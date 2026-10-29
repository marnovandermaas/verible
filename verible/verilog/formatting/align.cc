// Copyright 2017-2020 The Verible Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "verible/verilog/formatting/align.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <string_view>
#include <vector>

#include "absl/log/die_if_null.h"
#include "absl/strings/ascii.h"
#include "verible/common/formatting/align.h"
#include "verible/common/formatting/format-token.h"
#include "verible/common/formatting/token-partition-tree.h"
#include "verible/common/formatting/unwrapped-line.h"
#include "verible/common/strings/position.h"
#include "verible/common/text/concrete-syntax-leaf.h"
#include "verible/common/text/concrete-syntax-tree.h"
#include "verible/common/text/symbol.h"
#include "verible/common/text/token-info.h"
#include "verible/common/text/tree-context-visitor.h"
#include "verible/common/text/tree-utils.h"
#include "verible/common/util/logging.h"
#include "verible/common/util/value-saver.h"
#include "verible/verilog/CST/context-functions.h"
#include "verible/verilog/CST/declaration.h"
#include "verible/verilog/CST/verilog-nonterminals.h"
#include "verible/verilog/formatting/format-style.h"
#include "verible/verilog/parser/verilog-token-classifications.h"
#include "verible/verilog/parser/verilog-token-enum.h"

namespace verilog {
namespace formatter {

using verible::AlignablePartitionGroup;
using verible::AlignedPartitionClassification;
using verible::AlignmentCellScannerGenerator;
using verible::AlignmentColumnProperties;
using verible::AlignmentGroupAction;
using verible::ByteOffsetSet;
using verible::ColumnSchemaScanner;
using verible::ExtractAlignmentGroupsFunction;
using verible::FormatTokenRange;
using verible::PreFormatToken;
using verible::Symbol;
using verible::SyntaxTreeLeaf;
using verible::SyntaxTreeNode;
using verible::SyntaxTreePath;
using verible::TaggedTokenPartitionRange;
using verible::TokenPartitionRange;
using verible::TokenPartitionTree;
using verible::TreePathFormatter;
using verible::ValueSaver;

static constexpr AlignmentColumnProperties FlushLeft(true);
static constexpr AlignmentColumnProperties FlushRight(false);

static const SyntaxTreePath::value_type kLeadingNonTreeTokenPathIndex = -1;
static const SyntaxTreePath::value_type kTrailingNonTreeTokenPathIndex =
    std::numeric_limits<SyntaxTreePath::value_type>::max();

// Maximum SyntaxTreePath index available for tree tokens
static const SyntaxTreePath::value_type kMaxPathIndex =
    std::numeric_limits<SyntaxTreePath::value_type>::max() - 1;

template <class T>
static bool TokensAreAllCommentsOrAttributes(const T &tokens) {
  return std::all_of(
      tokens.begin(), tokens.end(), [](const typename T::value_type &token) {
        const auto tag = static_cast<verilog_tokentype>(token.TokenEnum());
        return IsComment(verilog_tokentype(tag)) ||
               tag == verilog_tokentype::TK_ATTRIBUTE;
      });
}

template <class T>
static bool TokensHaveParenthesis(const T &tokens) {
  return std::any_of(tokens.begin(), tokens.end(),
                     [](const typename T::value_type &token) {
                       return token.TokenEnum() == '(';
                     });
}

// Returns true if the partition contains a separator comment line.
// A separator comment is an EOL comment whose body contains a run of 4 or
// more consecutive identical "divider" characters, where a divider character
// is any non-alphanumeric, non-whitespace character (e.g. '-', '=', '*',
// '#', '/').  The run may be surrounded by other text, so both a bare rule
// and a captioned one are recognised.  Examples: "// ----", "// ====",
// "/////////////////////", "// ------ section heading ------".
static bool IsSeparatorComment(const TokenPartitionTree &partition) {
  const auto &uwline = partition.Value();
  const auto token_range = uwline.TokensRange();
  if (token_range.empty()) return false;

  constexpr int kMinRunLength = 4;
  for (const auto &ftoken : token_range) {
    if (ftoken.TokenEnum() !=
        static_cast<int>(verilog_tokentype::TK_EOL_COMMENT)) {
      continue;
    }

    std::string_view text = ftoken.Text();
    // Strip leading "//"
    if (text.size() < 2 || text[0] != '/' || text[1] != '/') continue;
    const std::string_view body = text.substr(2);

    // Look for a run of kMinRunLength or more consecutive identical divider
    // characters anywhere in the comment body.
    int run = 0;
    char prev = '\0';
    for (const char ch : body) {
      const bool is_divider =
          !absl::ascii_isalnum(ch) && !absl::ascii_isspace(ch);
      if (is_divider && ch == prev) {
        if (++run >= kMinRunLength) return true;
      } else {
        run = is_divider ? 1 : 0;
        prev = ch;
      }
    }
  }
  return false;
}

static bool BlankLinesBreakGroups(AlignmentGroupBoundary b) {
  return b == AlignmentGroupBoundary::kBlankLines ||
         b == AlignmentGroupBoundary::kBlankLinesAndSeparatorComments;
}

static bool SeparatorCommentsBreakGroups(AlignmentGroupBoundary b) {
  return b == AlignmentGroupBoundary::kSeparatorComments ||
         b == AlignmentGroupBoundary::kBlankLinesAndSeparatorComments;
}

// True when non-tree tokens (e.g. // comments, line-continuation `\`) precede
// the origin, and the first origin token must start a new line. Aligning such
// partitions would glue the leading tokens onto the origin line via kInline
// cells (GitHub issue 2539). Leave them out of alignment instead.
static bool PartitionHasLeadingTokensBeforeForcedWrap(
    const TokenPartitionTree &partition) {
  const auto &uwline = partition.Value();
  const verible::Symbol *origin = uwline.Origin();
  if (origin == nullptr) return false;

  const auto ftokens = uwline.TokensRange();
  if (ftokens.empty()) return false;

  const verible::SyntaxTreeLeaf *first_leaf = verible::GetLeftmostLeaf(*origin);
  if (first_leaf == nullptr) return false;

  const verible::TokenInfo &first_tree_token = first_leaf->get();
  auto ftoken_it = ftokens.begin();
  while (ftoken_it != ftokens.end() &&
         *(ftoken_it->token) != first_tree_token) {
    ++ftoken_it;
  }
  if (ftoken_it == ftokens.begin() || ftoken_it == ftokens.end()) return false;
  return ftoken_it->before.break_decision == verible::SpacingOptions::kMustWrap;
}

static bool IgnoreCommentsAndPreprocessingDirectives(
    const TokenPartitionTree &partition) {
  const auto &uwline = partition.Value();

  // ignore partitions with only non-tree tokens (comments, comma-only lines)
  if (!uwline.Origin()) return true;

  const auto token_range = uwline.TokensRange();
  CHECK(!token_range.empty());
  // ignore lines containing only comments
  if (TokensAreAllCommentsOrAttributes(token_range)) return true;

  if (PartitionHasLeadingTokensBeforeForcedWrap(partition)) return true;

  // ignore partitions belonging to preprocessing directives
  return IsPreprocessorKeyword(
      verilog_tokentype(token_range.front().TokenEnum()));
}

static bool IgnoreWithinPortDeclarationPartitionGroup(
    const TokenPartitionTree &partition) {
  const auto &uwline = partition.Value();
  const auto token_range = uwline.TokensRange();
  CHECK(!token_range.empty());
  if (IgnoreCommentsAndPreprocessingDirectives(partition)) return true;

  // Ignore .x or .x(x) port declarations.
  // These can appear in a list_of_port_or_port_declarations.
  CHECK_NOTNULL(uwline.Origin());
  return uwline.Origin()->Kind() == verible::SymbolKind::kNode &&
         verible::SymbolCastToNode(*uwline.Origin())
             .MatchesTag(NodeEnum::kPort);
}

static bool IgnoreWithinStructUnionMemberPartitionGroup(
    const TokenPartitionTree &partition) {
  const auto &uwline = partition.Value();
  const auto token_range = uwline.TokensRange();
  CHECK(!token_range.empty());
  // TODO(mglb): Verify whether `IgnoreCommentsAndPreprocessingDirectives` can
  // be used instead of a direct comments/preprocessor tests (like in other
  // IgnoreWithin* functions). If so, use it.

  // ignore lines containing only comments
  if (TokensAreAllCommentsOrAttributes(token_range)) {
    return true;
  }

  // ignore partitions belonging to preprocessing directives
  if (IsPreprocessorKeyword(
          verilog_tokentype(token_range.front().TokenEnum()))) {
    return true;
  }

  if (PartitionHasLeadingTokensBeforeForcedWrap(partition)) return true;

  // ignore nested structs/unions
  if (verible::FindFirstSubtree(
          partition.Value().Origin(), [](const Symbol &symbol) {
            return symbol.Tag() ==
                   verible::NodeTag(NodeEnum::kStructUnionMemberList);
          }) != nullptr) {
    return true;
  }

  return false;
}

static bool IgnoreWithinActualNamedParameterPartitionGroup(
    const TokenPartitionTree &partition) {
  if (IgnoreCommentsAndPreprocessingDirectives(partition)) return true;

  // ignore everything that isn't passing a parameter by name
  const auto &uwline = partition.Value();
  CHECK_NOTNULL(uwline.Origin());
  return !(uwline.Origin()->Kind() == verible::SymbolKind::kNode &&
           verible::SymbolCastToNode(*uwline.Origin())
               .MatchesTag(NodeEnum::kParamByName));
}

static bool IgnoreWithinActualNamedPortPartitionGroup(
    const TokenPartitionTree &partition) {
  if (IgnoreCommentsAndPreprocessingDirectives(partition)) return true;

  const auto &uwline = partition.Value();
  const auto token_range = uwline.TokensRange();

  // ignore wildcard connections .*
  if (verilog_tokentype(token_range.front().TokenEnum()) ==
      verilog_tokentype::TK_DOTSTAR) {
    return true;
  }

  CHECK_NOTNULL(uwline.Origin());
  if (uwline.Origin()->Kind() != verible::SymbolKind::kNode) return true;

  // ignore implicit connections .aaa
  if (verible::SymbolCastToNode(*uwline.Origin())
          .MatchesTag(NodeEnum::kActualNamedPort) &&
      !TokensHaveParenthesis(token_range)) {
    return true;
  }

  // ignore positional port connections
  if (verible::SymbolCastToNode(*uwline.Origin())
          .MatchesTag(NodeEnum::kActualPositionalPort)) {
    return true;
  }

  return false;
}

static bool TokenForcesLineBreak(const PreFormatToken &ftoken) {
  switch (ftoken.TokenEnum()) {
    case verilog_tokentype::TK_begin:
    case verilog_tokentype::TK_fork:
      return true;
    default:
      return false;
  }
}

// Returns true if a token after the first one of the partition forces a line
// break, e.g. after an EOL comment. Aligning such a partition as one row would
// join the comment and the code that follows it on one line.
static bool PartitionHasForcedLineBreak(const TokenPartitionTree &partition) {
  const auto token_range = partition.Value().TokensRange();
  if (token_range.empty()) return false;
  return std::any_of(
      token_range.begin() + 1, token_range.end(),
      [&token_range](const PreFormatToken &ftoken) {
        return ftoken.before.break_decision ==
                   verible::SpacingOptions::kMustWrap ||
               (ftoken.TokenEnum() == verilog_tokentype::TK_EOL_COMMENT &&
                &ftoken != &token_range.back());
      });
}

static bool IgnoreMultilineCaseStatements(const TokenPartitionTree &partition) {
  if (IgnoreCommentsAndPreprocessingDirectives(partition)) return true;

  const auto &uwline = partition.Value();
  const auto token_range = uwline.TokensRange();

  // Scan for any tokens that would force a line break.
  if (std::any_of(token_range.begin(), token_range.end(),
                  &TokenForcesLineBreak)) {
    return true;
  }

  // Comments inside the case item force line breaks too, e.g. an EOL comment
  // after the case label.
  return PartitionHasForcedLineBreak(partition);
}

// Ignores statements that span multiple lines: with a forced line break, or
// with a part that is always expanded, e.g. named constructor arguments laid
// out like port connections.
static bool HasAlwaysExpandedSubpartition(const TokenPartitionTree &partition) {
  for (const auto &child : partition.Children()) {
    if ((child.Children().size() > 1 &&
         child.Value().PartitionPolicy() ==
             verible::PartitionPolicyEnum::kAlwaysExpand) ||
        HasAlwaysExpandedSubpartition(child)) {
      return true;
    }
  }
  return false;
}

static bool IgnoreMultilineStatements(const TokenPartitionTree &partition) {
  if (IgnoreCommentsAndPreprocessingDirectives(partition)) return true;
  return PartitionHasForcedLineBreak(partition) ||
         HasAlwaysExpandedSubpartition(partition);
}

class VerilogColumnSchemaScanner : public ColumnSchemaScanner {
 public:
  explicit VerilogColumnSchemaScanner(const FormatStyle &style)
      : style_(style) {}

 protected:
  const FormatStyle &style_;
};

template <class ScannerType>
static std::function<verible::AlignmentCellScannerFunction(const FormatStyle &)>
UnstyledAlignmentCellScannerGenerator() {
  return [](const FormatStyle &vstyle) {
    return AlignmentCellScannerGenerator<ScannerType>(
        [vstyle] { return ScannerType(vstyle); });
  };
}

// Wraps a scanner of non-tree tokens to mark its trailing comment columns as
// aligned by run, see FormatStyle::align_trailing_comments_by_run.
static verible::NonTreeTokensScannerFunction TrailingCommentsAlignedByRun(
    const verible::NonTreeTokensScannerFunction &non_tree_column_scanner) {
  return [non_tree_column_scanner](
             verible::FormatTokenRange leading_tokens,
             verible::FormatTokenRange trailing_tokens,
             verible::ColumnPositionTree *column_entries) {
    const size_t first_new = column_entries->Children().size();
    non_tree_column_scanner(leading_tokens, trailing_tokens, column_entries);
    for (size_t i = first_new; i < column_entries->Children().size(); ++i) {
      auto &column = column_entries->Children()[i].Value();
      const int token = column.starting_token.token_enum();
      if (token == TK_EOL_COMMENT || token == TK_COMMENT_BLOCK) {
        column.properties.aligned_by_run = true;
      }
    }
  };
}

template <class ScannerType>
std::function<verible::AlignmentCellScannerFunction(
    const FormatStyle
        &)> static UnstyledAlignmentCellScannerGenerator(const verible::NonTreeTokensScannerFunction
                                                             &non_tree_column_scanner) {
  return [non_tree_column_scanner](const FormatStyle &vstyle) {
    return AlignmentCellScannerGenerator<ScannerType>(
        [vstyle] { return ScannerType(vstyle); },
        vstyle.align_trailing_comments_by_run
            ? TrailingCommentsAlignedByRun(non_tree_column_scanner)
            : non_tree_column_scanner);
  };
}

// This class marks up token-subranges in named parameter assignments for
// alignment. e.g. ".parameter_name(value_expression)"
class ActualNamedParameterColumnSchemaScanner
    : public VerilogColumnSchemaScanner {
 public:
  explicit ActualNamedParameterColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());
    // The scanner starts at the aligned row's kParamByName. Columns are only
    // created for it, not for named arguments of function calls nested in its
    // value, e.g. ".Depth (f(.x(1)))".
    switch (tag) {
      case NodeEnum::kParamByName: {
        // Always start first column right away
        if (!Context().IsInside(NodeEnum::kParamByName)) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
      }
      case NodeEnum::kParenGroup:
        // Second column starts at the open parenthesis.
        if (Context().DirectParentIs(NodeEnum::kParamByName) &&
            !Context().IsInsideStartingFrom(NodeEnum::kParamByName, 1)) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
      default:
        // With padded parentheses, the value gets its own column so that the
        // closing parentheses line up: ".a   ( 1    )".
        if (InPaddedParamParens()) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
    }
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    // The value (when it is a single token) and the closing parenthesis start
    // new columns. The open parenthesis stays in the paren group's column.
    if (!InPaddedParamParens()) return;
    const int token = leaf.get().token_enum();
    if (token == '(') return;
    const SyntaxTreeNode &paren_group = Context().top();
    if (token == ')' && paren_group.size() == 3 && paren_group[1] == nullptr) {
      // Empty "( )": put ')' in the value column, so that it neither widens
      // the '(' column nor is pushed out to align with the other ')'.
      SyntaxTreePath value_path = Path();
      value_path.back() = 1;
      ReserveNewColumn(leaf, FlushLeft, value_path);
      return;
    }
    ReserveNewColumn(leaf, FlushLeft);
  }

 private:
  // True for direct children of the parentheses of the row's named parameter
  // assignment, when these are padded.
  bool InPaddedParamParens() const {
    return style_.named_parameter_paren_padding &&
           Context().DirectParentsAre(
               {NodeEnum::kParenGroup, NodeEnum::kParamByName}) &&
           !Context().IsInsideStartingFrom(NodeEnum::kParamByName, 2);
  }
};

// This class marks up token-subranges in named arguments of class constructor
// calls for alignment, like named port connections: ".name (value)".
class NamedConstructorArgumentColumnSchemaScanner
    : public VerilogColumnSchemaScanner {
 public:
  explicit NamedConstructorArgumentColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    // The scanner starts at the row's kParamByName. Columns are only created
    // for it, not for named arguments of calls nested in its value.
    switch (NodeEnum(node.Tag().tag)) {
      case NodeEnum::kParamByName:
        if (!Context().IsInside(NodeEnum::kParamByName)) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
      case NodeEnum::kParenGroup:
        // Second column starts at the open parenthesis.
        if (Context().DirectParentIs(NodeEnum::kParamByName) &&
            !Context().IsInsideStartingFrom(NodeEnum::kParamByName, 1)) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
      default:
        break;
    }
    TreeContextPathVisitor::Visit(node);
  }
};

// This class marks up token-subranges in single-line typedefs for alignment:
// "typedef <type> <name> [<unpacked dimensions>];", aligning the names.
class TypedefColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit TypedefColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    // The row's origin: first column, "typedef <type>".
    if (Context().empty() && node.MatchesTag(NodeEnum::kTypeDeclaration)) {
      ReserveNewColumn(node, FlushLeft);
    }
    TreeContextPathVisitor::Visit(node);
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    // The declared name, a direct child after the type: second column.
    if (Context().size() == 1 &&
        Context().DirectParentIs(NodeEnum::kTypeDeclaration) &&
        Path() == SyntaxTreePath{2}) {
      ReserveNewColumn(leaf, FlushLeft);
    }
  }
};

// This class marks up token-subranges in named port connections for alignment.
// e.g. ".port_name(net_name)"
class ActualNamedPortColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit ActualNamedPortColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());
    switch (tag) {
      case NodeEnum::kActualNamedPort: {
        // Always start first column right away
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      case NodeEnum::kParenGroup:
        // Second column starts at the open parenthesis.
        if (Context().DirectParentIs(NodeEnum::kActualNamedPort)) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
      default:
        break;
    }
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }
};

// This class marks up token-subranges in port declarations for alignment.
// e.g. "input wire clk,"
class PortDeclarationColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit PortDeclarationColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());
    switch (tag) {
      case NodeEnum::kPackedDimensions: {
        // Kludge: kPackedDimensions can appear in paths
        //   [1,0,3] inside a kNetDeclaration and at
        //   [1,0,0,3] inside a kDataDeclaration,
        // but we want them to line up in the same column.  Make it so.
        // TODO(fangism): a swap-based saver would be more efficient
        // for vectors.

        SyntaxTreePath new_path;
        if (current_path_ == SyntaxTreePath{1, 0, 3}) {
          new_path = {1, 0, 0, 3};
        } else {
          new_path = Path();
        }
        const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);

        // Left border is removed from each dimension subcolumn.
        // Adding it here creates one space before first column.
        static const verible::AlignmentColumnProperties single_left_border(true,
                                                                           1);

        current_dimensions_group_ = ReserveNewColumn(node, single_left_border);
        TreeContextPathVisitor::Visit(node);
        current_dimensions_group_ = nullptr;
        return;
      }
      case NodeEnum::kUnpackedDimensions: {
        current_dimensions_group_ = ReserveNewColumn(node, FlushLeft);
        TreeContextPathVisitor::Visit(node);
        current_dimensions_group_ = nullptr;
        return;
      }
      case NodeEnum::kDimensionRange:
      case NodeEnum::kDimensionSlice: {
        CHECK_NOTNULL(current_dimensions_group_);
        CHECK_EQ(node.size(), 5);

        SyntaxTreePath dimension_path = Path();
        const bool right_align =
            Context().IsInside(NodeEnum::kPackedDimensions)
                ? style_.port_declarations_right_align_packed_dimensions
                : style_.port_declarations_right_align_unpacked_dimensions;
        if (right_align) {
          dimension_path.back() += kMaxPathIndex - Context().top().size();
        }

        const verible::AlignmentColumnProperties no_border(false, 0);
        auto *column = ABSL_DIE_IF_NULL(ReserveNewColumn(
            current_dimensions_group_, node,
            right_align ? no_border : FlushLeft, dimension_path));

        ReserveNewColumn(column, *node[0],
                         right_align ? no_border : FlushLeft);  // '['
        ReserveNewColumn(column, *node[1], FlushRight);         // value
        ReserveNewColumn(column, *node[4], FlushLeft);          // ']'
        return;
      }
      case NodeEnum::kDimensionScalar:
      case NodeEnum::kDimensionAssociativeType: {
        CHECK_NOTNULL(current_dimensions_group_);
        CHECK_EQ(node.size(), 3);

        SyntaxTreePath dimension_path = Path();
        const bool right_align =
            Context().IsInside(NodeEnum::kPackedDimensions)
                ? style_.port_declarations_right_align_packed_dimensions
                : style_.port_declarations_right_align_unpacked_dimensions;
        if (right_align) {
          dimension_path.back() += kMaxPathIndex - Context().top().size();
        }

        const verible::AlignmentColumnProperties no_border(false, 0);

        auto *column = ABSL_DIE_IF_NULL(ReserveNewColumn(
            current_dimensions_group_, node,
            right_align ? no_border : FlushLeft, dimension_path));

        const auto &column_path = column->Value().path;
        // Value can be empty - set paths explicitly
        ReserveNewColumn(column, *node[0], right_align ? no_border : FlushLeft,
                         GetSubpath(column_path, {0}));  // '['
        ReserveNewColumn(column, *node[1], FlushRight,
                         GetSubpath(column_path, {1}));  // value
        ReserveNewColumn(column, *node[2], FlushLeft,
                         GetSubpath(column_path, {2}));  // ']'
        return;
      }

      case NodeEnum::kDataType:
        // appears in path [2,0]
        ReserveNewColumn(node, FlushLeft);
        break;

      case NodeEnum::kUnqualifiedId:
        if (Context().DirectParentIs(NodeEnum::kPortDeclaration) ||
            Context().DirectParentsAre(
                {NodeEnum::kDataTypeImplicitBasicIdDimensions,
                 NodeEnum::kPortItem})) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
      case NodeEnum::kExpression:
        // optional: Early termination of tree traversal.
        // This also helps reduce noise during debugging of this visitor.
        return;
        // case NodeEnum::kConstRef: possible in CST, but should be
        // syntactically illegal in module ports context.
      default:
        break;
    }
    // recursive visitation
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());
    const int tag = leaf.get().token_enum();
    switch (tag) {
      // port directions
      case verilog_tokentype::TK_inout:
      case verilog_tokentype::TK_input:
      case verilog_tokentype::TK_output:
      case verilog_tokentype::TK_ref: {
        ReserveNewColumn(leaf, FlushLeft);
        break;
      }

      // net types
      case verilog_tokentype::TK_wire:
      case verilog_tokentype::TK_tri:
      case verilog_tokentype::TK_tri1:
      case verilog_tokentype::TK_supply0:
      case verilog_tokentype::TK_wand:
      case verilog_tokentype::TK_triand:
      case verilog_tokentype::TK_tri0:
      case verilog_tokentype::TK_supply1:
      case verilog_tokentype::TK_wor:
      case verilog_tokentype::TK_trior:
      case verilog_tokentype::TK_wone:
      case verilog_tokentype::TK_uwire: {
        // Effectively merge/re-map this into the next node slot,
        // which is kDataType of kPortDeclaration.
        // This works-around a quirk in the CST construction where net_types
        // like 'wire' appear positionally before kDataType variable types
        // like 'reg'.
        ReserveNewColumn(leaf, FlushLeft, verible::NextSiblingPath(Path()));
        break;
      }
      // TODO(b/70310743): Treat "[...:...]" as 5 columns.
      // Treat "[...]" (scalar) as 3 columns.
      // TODO(b/70310743): Treat the ... as a multi-column cell w.r.t.
      // the 5-column range format.
      default:
        break;
    }
    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }

 private:
  verible::ColumnPositionTree *current_dimensions_group_ = nullptr;
};

// This class marks up token-subranges in struct/union members for alignment.
// e.g. bit [31:0] member_name;
class StructUnionMemberColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit StructUnionMemberColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());
    switch (tag) {
      case NodeEnum::kStructUnionMember: {
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      case NodeEnum::kTrailingAssign: {
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      case NodeEnum::kVariableDeclarationAssignmentList:
      case NodeEnum::kVariableDeclarationAssignment:
      case NodeEnum::kDataTypeImplicitIdDimensions:
        break;

      default:
        return;
    }
    // recursive visitation
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());
    const int tag = leaf.get().token_enum();
    switch (tag) {
      case verilog_tokentype::SymbolIdentifier:
      case verilog_tokentype::EscapedIdentifier: {
        // Member ID in kDataTypeImplicitIdDimensions can be at [1] or [2].
        if (current_path_ == SyntaxTreePath{1, 1}) {
          SyntaxTreePath new_path{1, 2};
          const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
          ReserveNewColumn(leaf, FlushLeft);
        } else {
          ReserveNewColumn(leaf, FlushLeft);
        }
        break;
      }
      default:
        break;
    }
    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }
};

static bool IsAlignableDeclaration(const SyntaxTreeNode &node) {
  // A data/net/variable declaration is alignable if:
  // * it is not a module instance
  // * it declares exactly one identifier
  switch (static_cast<NodeEnum>(node.Tag().tag)) {
    case NodeEnum::kDataDeclaration: {
      const SyntaxTreeNode *instances(GetInstanceListFromDataDeclaration(node));
      if (!instances) return false;
      if (FindAllRegisterVariables(*instances).size() > 1) return false;
      return FindAllGateInstances(*instances).empty();
    }
    case NodeEnum::kNetDeclaration: {
      return FindAllNetVariables(node).size() <= 1;
    }
    default:
      return false;
  }
}

// These enums classify alignable groups of token partitions by their syntax
// structure, which then map to different alignment handler routines.
// These need not have a 1:1 correspondence to verilog::NodeEnum syntax tree
// enums, a single value here could apply to a group of syntax tree node types.
enum class AlignableSyntaxSubtype {
  kDontCare = 0,
  kNamedActualParameters,
  kNamedActualPorts,
  kParameterDeclaration,      // formal parameter list (#(...))
  kBodyParameterDeclaration,  // parameter/localparam in module/generate/
                              //   package/interface body
  kPortDeclaration,
  kStructUnionMember,
  kDataDeclaration,  // net/variable declarations
  kClassMemberVariables,
  kCaseLikeItems,
  kContinuousAssignment,
  kEnumListAssignment,  // Constants aligned in enums.
  kBlockingAssignment,
  kNonBlockingAssignment,
  kDistItem,  // Distribution items.
  kIfElseBranches,  // Branches of if/else-if/else chains.
  kNamedConstructorArguments,  // Named arguments of new(...).
  kTypedef,                    // Single-line typedefs.
};

static AlignedPartitionClassification AlignClassify(
    AlignmentGroupAction match,
    AlignableSyntaxSubtype subtype = AlignableSyntaxSubtype::kDontCare) {
  if (match == AlignmentGroupAction::kMatch) {
    CHECK(subtype != AlignableSyntaxSubtype::kDontCare);
  }
  return {match, static_cast<int>(subtype)};
}

// Returns true for single-line typedefs without a body, which are aligned as
// kTypedef rows. Typedefs of enums, structs and unions break the group.
static bool IsAlignableTypedef(const TokenPartitionTree &partition,
                               const SyntaxTreeNode &node) {
  if (!node.MatchesTag(NodeEnum::kTypeDeclaration)) return false;
  const auto tokens = partition.Value().TokensRange();
  return std::none_of(tokens.begin(), tokens.end(),
                      [](const PreFormatToken &t) {
                        return t.TokenEnum() == '{';
                      }) &&
         !PartitionHasForcedLineBreak(partition);
}

static std::vector<TaggedTokenPartitionRange> GetConsecutiveModuleItemGroups(
    const TokenPartitionRange &partitions, AlignmentGroupBoundary boundary) {
  VLOG(2) << __FUNCTION__;
  return GetPartitionAlignmentSubranges(
      partitions,  //
      [boundary](const TokenPartitionTree &partition)
          -> AlignedPartitionClassification {
        const Symbol *origin = partition.Value().Origin();
        if (origin == nullptr) {
          if (SeparatorCommentsBreakGroups(boundary) &&
              IsSeparatorComment(partition)) {
            return {AlignmentGroupAction::kNoMatch};
          }
          return {AlignmentGroupAction::kIgnore};
        }
        const verible::SymbolTag symbol_tag = origin->Tag();
        if (symbol_tag.kind != verible::SymbolKind::kNode) {
          return AlignClassify(AlignmentGroupAction::kIgnore);
        }
        const SyntaxTreeNode &node = verible::SymbolCastToNode(*origin);
        if (IsAlignableTypedef(partition, node)) {
          return AlignClassify(AlignmentGroupAction::kMatch,
                               AlignableSyntaxSubtype::kTypedef);
        }
        // Align body-level parameter/localparam declarations.
        if (node.MatchesTag(NodeEnum::kParamDeclaration)) {
          return AlignClassify(
              AlignmentGroupAction::kMatch,
              AlignableSyntaxSubtype::kBodyParameterDeclaration);
        }

        // Align net/variable declarations.
        if (IsAlignableDeclaration(node)) {
          return AlignClassify(AlignmentGroupAction::kMatch,
                               AlignableSyntaxSubtype::kDataDeclaration);
        }
        // Align continuous assignment, like "assign foo = bar;"
        if (node.MatchesTag(NodeEnum::kContinuousAssignmentStatement)) {
          return AlignClassify(AlignmentGroupAction::kMatch,
                               AlignableSyntaxSubtype::kContinuousAssignment);
        }
        return AlignClassify(AlignmentGroupAction::kNoMatch);
      });
}

static std::vector<TaggedTokenPartitionRange> GetConsecutiveClassItemGroups(
    const TokenPartitionRange &partitions, AlignmentGroupBoundary boundary,
    bool align_parameter_declarations) {
  VLOG(2) << __FUNCTION__;
  return GetPartitionAlignmentSubranges(
      partitions,  //
      [boundary, align_parameter_declarations](
          const TokenPartitionTree &partition)
          -> AlignedPartitionClassification {
        const Symbol *origin = partition.Value().Origin();
        if (origin == nullptr) {
          if (SeparatorCommentsBreakGroups(boundary) &&
              IsSeparatorComment(partition)) {
            return {AlignmentGroupAction::kNoMatch};
          }
          return {AlignmentGroupAction::kIgnore};
        }
        const verible::SymbolTag symbol_tag = origin->Tag();
        if (symbol_tag.kind != verible::SymbolKind::kNode) {
          return {AlignmentGroupAction::kIgnore};
        }
        const SyntaxTreeNode &node = verible::SymbolCastToNode(*origin);
        if (IsAlignableTypedef(partition, node)) {
          return AlignClassify(AlignmentGroupAction::kMatch,
                               AlignableSyntaxSubtype::kTypedef);
        }
        // Align parameter/localparam declarations, like in module bodies.
        if (align_parameter_declarations &&
            node.MatchesTag(NodeEnum::kParamDeclaration)) {
          return AlignClassify(
              AlignmentGroupAction::kMatch,
              AlignableSyntaxSubtype::kBodyParameterDeclaration);
        }
        // Align class member variables.
        return AlignClassify(IsAlignableDeclaration(node)
                                 ? AlignmentGroupAction::kMatch
                                 : AlignmentGroupAction::kNoMatch,
                             AlignableSyntaxSubtype::kClassMemberVariables);
      });
}

static std::vector<TaggedTokenPartitionRange> GetAlignableStatementGroups(
    const TokenPartitionRange &partitions, AlignmentGroupBoundary boundary) {
  VLOG(2) << __FUNCTION__;
  return GetPartitionAlignmentSubranges(
      partitions,  //
      [boundary](const TokenPartitionTree &partition)
          -> AlignedPartitionClassification {
        const Symbol *origin = partition.Value().Origin();
        if (origin == nullptr) {
          if (SeparatorCommentsBreakGroups(boundary) &&
              IsSeparatorComment(partition)) {
            return {AlignmentGroupAction::kNoMatch};
          }
          return {AlignmentGroupAction::kIgnore};
        }
        const verible::SymbolTag symbol_tag = origin->Tag();
        if (symbol_tag.kind != verible::SymbolKind::kNode) {
          return AlignClassify(AlignmentGroupAction::kIgnore);
        }
        const SyntaxTreeNode &node = verible::SymbolCastToNode(*origin);
        // Align local variable declarations.
        if (IsAlignableDeclaration(node)) {
          return AlignClassify(AlignmentGroupAction::kMatch,
                               AlignableSyntaxSubtype::kDataDeclaration);
        }
        // Align blocking assignments.
        if (node.MatchesTagAnyOf({NodeEnum::kBlockingAssignmentStatement,
                                  NodeEnum::kNetVariableAssignment})) {
          return AlignClassify(AlignmentGroupAction::kMatch,
                               AlignableSyntaxSubtype::kBlockingAssignment);
        }
        // Align nonblocking assignments.
        if (node.MatchesTag(NodeEnum::kNonblockingAssignmentStatement)) {
          return AlignClassify(AlignmentGroupAction::kMatch,
                               AlignableSyntaxSubtype::kNonBlockingAssignment);
        }
        return AlignClassify(AlignmentGroupAction::kNoMatch);
      });
}

// This class marks up token-subranges in data declarations for alignment.
// e.g. "foo_pkg::bar_t [3:0] some_values;"
// Much of the implementation of this scanner was based on
// PortDeclarationColumnSchemaScanner.
// Differences:
//   * here, there are no port directions to worry about.
//   * need to handle both kDataDeclaration and kNetDeclaration.
// TODO(fangism): refactor out common logic
class DataDeclarationColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit DataDeclarationColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());
    if (new_column_after_open_bracket_) {
      ReserveNewColumn(node, FlushRight);
      new_column_after_open_bracket_ = false;
      TreeContextPathVisitor::Visit(node);
      return;
    }
    switch (tag) {
      case NodeEnum::kDataDeclaration:
      case NodeEnum::kNetDeclaration: {
        // Don't wait for the type node, just start the first column right away.
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      case NodeEnum::kPackedDimensions: {
        // Kludge: kPackedDimensions can appear in paths:
        //   [1,0,3] inside a kNetDeclaration and at
        //   [1,0,0,3] inside a kDataDeclaration,
        // but we want them to line up in the same column.  Make it so.
        if (current_path_ == SyntaxTreePath{1, 0, 0, 3}) {
          SyntaxTreePath new_path{1, 0, 3};
          const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
          TreeContextPathVisitor::Visit(node);
          return;
        }
        break;
      }
      case NodeEnum::kDeclarationDimensions: {
        if (current_path_ == SyntaxTreePath{1, 0, 3, 0}) {
          SyntaxTreePath new_path{1, 0, 3};
          const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
          TreeContextPathVisitor::Visit(node);
          return;
        }
        break;
      }
      case NodeEnum::kDimensionScalar: {
        CHECK_EQ(node.size(), 3);
        auto *column = ABSL_DIE_IF_NULL(ReserveNewColumn(node, FlushLeft));

        ReserveNewColumn(column, *node[0], FlushLeft);   // '['
        ReserveNewColumn(column, *node[1], FlushRight);  // value
        ReserveNewColumn(column, *node[2], FlushLeft);   // ']'
        return;
      }
      case NodeEnum::kDimensionRange: {
        CHECK_EQ(node.size(), 5);
        auto *column = ABSL_DIE_IF_NULL(ReserveNewColumn(node, FlushRight));

        // The value returned from this call can be used to
        // center the ranges instead of right-align which
        // was the correct way according to tests
        ReserveNewColumn(column, *node[0], FlushLeft);  // '['

        ReserveNewColumn(column, *node[1], FlushRight);
        ReserveNewColumn(*node[1], FlushRight);  // LHS value
        ReserveNewColumn(*node[2], FlushLeft);   // ':'
        ReserveNewColumn(*node[3], FlushLeft);   // RHS value

        ReserveNewColumn(column, *node[4], FlushLeft);  // ']'
        return;
      }
      case NodeEnum::kDimensionSlice:
      case NodeEnum::kDimensionAssociativeType: {
        // All of these cases cover packed and unpacked dimensions
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      case NodeEnum::kRegisterVariable: {
        // at path [1,1,0] in kDataDeclaration
        // contains the declared id
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      case NodeEnum::kNetDeclarationAssignment:
      case NodeEnum::kNetVariable: {
        // at path [2,0] in kNetDeclaration
        // contains the declared id
        // make this fit with kRegisterVariable
        if (current_path_ == SyntaxTreePath{2, 0}) {
          SyntaxTreePath new_path{1, 1, 0};
          const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
          ReserveNewColumn(node, FlushLeft);
          TreeContextPathVisitor::Visit(node);
          return;
        }
        break;
      }
      case NodeEnum::kExpression:
        // optional: Early termination of tree traversal.
        // This also helps reduce noise during debugging of this visitor.
        return;
      // case NodeEnum::kConstRef: possible in CST, but should be
      // syntactically illegal in module ports context.
      default:
        break;
    }
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << "end of " << __FUNCTION__ << ", node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());
    if (new_column_after_open_bracket_) {
      ReserveNewColumn(leaf, FlushRight);
      new_column_after_open_bracket_ = false;
      return;
    }
    const int tag = leaf.get().token_enum();
    switch (tag) {
      // TODO(b/70310743): Treat "[...:...]" as 5 columns.
      // Treat "[...]" (scalar) as 3 columns.
      // TODO(b/70310743): Treat the ... as a multi-column cell w.r.t.
      // the 5-column range format.
      default:
        break;
    }
    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }

 protected:
  bool ContextAtDeclarationDimensions() const {
    // Alternatively, could check that grandparent is
    // kDeclarationDimensions.
    return current_context_.DirectParentIsOneOf(
        {NodeEnum::kDimensionRange, NodeEnum::kDimensionScalar,
         NodeEnum::kDimensionSlice, NodeEnum::kDimensionAssociativeType});
  }

 private:
  // Set this to force the next syntax tree node/leaf to start a new column.
  // This is useful for aligning after punctation marks.
  bool new_column_after_open_bracket_ = false;
};

// This class marks up token-subranges in class member variable (data
// declarations) for alignment. e.g. "const int [3:0] member_name;" For now,
// re-use the same column scanner as data/variable/net declarations.
class ClassPropertyColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit ClassPropertyColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());
    switch (tag) {
      case NodeEnum::kDeclarationDimensions: {
        if (current_path_ == SyntaxTreePath{1, 0, 0, 3, 0}) {
          SyntaxTreePath new_path{1, 0, 0, 3};
          const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
          TreeContextPathVisitor::Visit(node);
          return;
        }
        break;
      }
      case NodeEnum::kDataDeclaration:
      case NodeEnum::kVariableDeclarationAssignment: {
        // Don't wait for the type node, just start the first column right away.
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      case NodeEnum::kDimensionScalar: {
        CHECK_EQ(node.size(), 3);
        auto *column = ABSL_DIE_IF_NULL(ReserveNewColumn(node, FlushLeft));

        ReserveNewColumn(column, *node[0], FlushLeft);   // '['
        ReserveNewColumn(column, *node[1], FlushRight);  // value
        ReserveNewColumn(column, *node[2], FlushLeft);   // ']'
        return;
      }
      case NodeEnum::kDimensionRange: {
        CHECK_EQ(node.size(), 5);
        auto *column = ABSL_DIE_IF_NULL(ReserveNewColumn(node, FlushLeft));

        ReserveNewColumn(column, *node[0], FlushLeft);  // '['

        auto *value_subcolumn =
            ABSL_DIE_IF_NULL(ReserveNewColumn(column, *node[1], FlushRight));
        ReserveNewColumn(value_subcolumn, *node[1], FlushRight);  // LHS value
        ReserveNewColumn(value_subcolumn, *node[2], FlushLeft);   // ':'
        ReserveNewColumn(value_subcolumn, *node[3], FlushRight);  // RHS value

        ReserveNewColumn(column, *node[4], FlushLeft);  // ']'
        return;
      }
      default:
        break;
    }
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << "end of " << __FUNCTION__ << ", node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());
    const int tag = leaf.get().token_enum();
    switch (tag) {
      case '=':
        ReserveNewColumn(leaf, FlushLeft);
        break;
      default:
        break;
    }
    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }
};

// This class marks up token-subranges in formal and body-level parameter
// declarations for alignment.  e.g. "localparam int Width = 5;"
class ParameterDeclarationColumnSchemaScanner
    : public VerilogColumnSchemaScanner {
 public:
  explicit ParameterDeclarationColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());
    if (new_column_after_open_bracket_) {
      ReserveNewColumn(node, FlushRight);
      new_column_after_open_bracket_ = false;
      TreeContextPathVisitor::Visit(node);
      return;
    }

    switch (tag) {
      case NodeEnum::kTypeInfo: {
        SyntaxTreePath new_path{1};
        const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
        ReserveNewColumn(node, FlushLeft);
        break;
      }

      case NodeEnum::kTrailingAssign: {
        ReserveNewColumn(node, FlushLeft);
        break;
      }

      case NodeEnum::kUnqualifiedId: {
        if (Context().DirectParentIs(NodeEnum::kParamType)) {
          ReserveNewColumn(node, FlushLeft);
        }
        break;
      }
      default:
        break;
    }

    // recursive visitation
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());

    if (new_column_after_open_bracket_) {
      ReserveNewColumn(leaf, FlushRight);
      new_column_after_open_bracket_ = false;
      return;
    }

    const int tag = leaf.get().token_enum();
    switch (tag) {
      // Align keywords 'parameter', 'localparam' and 'type' under the same
      // column.
      case verilog_tokentype::TK_parameter:
      case verilog_tokentype::TK_localparam: {
        ReserveNewColumn(leaf, FlushLeft);
        break;
      }

      case verilog_tokentype::TK_type: {
        if (Context().DirectParentIs(NodeEnum::kParamDeclaration)) {
          ReserveNewColumn(leaf, FlushLeft);
        }
        break;
      }

      // Sometimes the parameter identifier which is of token SymbolIdentifier
      // can appear at different paths depending on the parameter type. Make
      // them aligned so they fall under the same column.
      case verilog_tokentype::SymbolIdentifier: {
        if (current_path_ == SyntaxTreePath{2, 0}) {
          SyntaxTreePath new_path{1, 2};
          const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
          ReserveNewColumn(leaf, FlushLeft);
          return;
        }

        if (Context().DirectParentIs(NodeEnum::kParamType)) {
          ReserveNewColumn(leaf, FlushLeft);
        }
        break;
      }

      // '=' is another column where things should be aligned. But type
      // declarations and localparam cause '=' to appear under two different
      // paths in CST. Align them.
      case '=': {
        if (current_path_ == SyntaxTreePath{2, 1}) {
          SyntaxTreePath new_path{2};
          const ValueSaver<SyntaxTreePath> path_saver(&current_path_, new_path);
          ReserveNewColumn(leaf, FlushLeft);
        }
        break;
      }

      // Align packed and unpacked dimensions
      case '[': {
        if (style_.parameter_declarations_align_dimensions &&
            verilog::analysis::ContextIsInsideDeclarationDimensions(
                Context()) &&
            !Context().IsInside(NodeEnum::kActualParameterList)) {
          // FlushLeft vs. Right doesn't matter, this is a single character.
          ReserveNewColumn(leaf, FlushLeft);
          new_column_after_open_bracket_ = true;
        }
        break;
      }
      case ']': {
        if (style_.parameter_declarations_align_dimensions &&
            verilog::analysis::ContextIsInsideDeclarationDimensions(
                Context()) &&
            !Context().IsInside(NodeEnum::kActualParameterList)) {
          // FlushLeft vs. Right doesn't matter, this is a single character.
          ReserveNewColumn(leaf, FlushLeft);
        }
        break;
      }

      default:
        break;
    }
    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }

 private:
  bool new_column_after_open_bracket_ = false;
};

// This class marks up token-subranges in case items for alignment.
// e.g. "value1, value2: x = f(y);"
// This is suitable for a variety of case-like items: statements, generate
// items.
class CaseItemColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit CaseItemColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

  bool ParentContextIsCaseItem() const {
    return Context().DirectParentIsOneOf(
        {NodeEnum::kCaseItem, NodeEnum::kCaseInsideItem,
         NodeEnum::kGenerateCaseItem, NodeEnum::kDefaultItem});
  }

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());

    if (previous_token_was_case_colon_) {
      if (ParentContextIsCaseItem()) {
        ReserveNewColumn(node, FlushLeft);
        previous_token_was_case_colon_ = false;
      }
    } else {
      switch (tag) {
        case NodeEnum::kCaseItem:
        case NodeEnum::kCaseInsideItem:
        case NodeEnum::kGenerateCaseItem:
        case NodeEnum::kDefaultItem: {
          // Start a new column right away.
          ReserveNewColumn(node, FlushLeft);
          break;
        }
        default:
          break;
      }
    }

    // recursive visitation
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());
    const int tag = leaf.get().token_enum();
    switch (tag) {
      case ':':
        if (ParentContextIsCaseItem()) {
          // mark the next node as the start of a new column
          previous_token_was_case_colon_ = true;
        }
        break;
      default:
        break;
    }
    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }

 private:
  bool previous_token_was_case_colon_ = false;
};

// This class marks up token-subranges in various assignment statements for
// alignment.  e.g.
// * assign foo = bar;
// * foo = bar;
// * foo <= bar;
class AssignmentColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit AssignmentColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());

    switch (tag) {
      case NodeEnum::kNetVariableAssignment:
      case NodeEnum::kBlockingAssignmentStatement:
      case NodeEnum::kNonblockingAssignmentStatement:
      case NodeEnum::kContinuousAssignmentStatement: {
        // Start a new column right away.
        ReserveNewColumn(node, FlushLeft);
        break;
      }
      default:
        break;
    }

    // recursive visitation
    TreeContextPathVisitor::Visit(node);
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());
    const int tag = leaf.get().token_enum();
    switch (tag) {
      case '=':  // align at '='
        if (Context().DirectParentIsOneOf(
                {NodeEnum::kNetVariableAssignment,
                 NodeEnum::kBlockingAssignmentStatement})) {
          ReserveNewColumn(leaf, FlushLeft);
        }
        break;
      case verilog_tokentype::TK_LE:  // '<=' for nonblocking assignments
        if (Context().DirectParentIs(
                NodeEnum::kNonblockingAssignmentStatement)) {
          ReserveNewColumn(leaf, FlushLeft);
        }
        break;
      default:
        break;
    }
    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }
};

// Aligns enums that have assignment.
// enum {       // cols:
//   foo = 42   // foo: flush left | =: left | ...: (default left)
// }
class EnumWithAssignmentsColumnSchemaScanner
    : public VerilogColumnSchemaScanner {
 public:
  explicit EnumWithAssignmentsColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    auto tag = NodeEnum(node.Tag().tag);
    VLOG(2) << __FUNCTION__ << ", node: " << tag << " at "
            << TreePathFormatter(Path());

    switch (tag) {
      case NodeEnum::kEnumName:
        ReserveNewColumn(node, FlushLeft);
        break;

      default: {
        // TODO(hzeller): Add third column for the assignment expression and
        // make (configurably?) align right if all of them are numbers ?
        // Need to keep track in little state machine, and across rows, as the
        // right side can also be any expression. If there is any expression,
        // we'd want to keep all left aligned. (nested TODO: keep state across
        // multiple rows ?)
      }
    }

    TreeContextPathVisitor::Visit(node);  // Recurse down.
    VLOG(2) << __FUNCTION__ << ", leaving node: " << tag;
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    VLOG(2) << __FUNCTION__ << ", leaf: " << leaf.get() << " at "
            << TreePathFormatter(Path());

    // Make sure that we only catch an = at the expected point
    if (Context().DirectParentIs(NodeEnum::kTrailingAssign) &&
        leaf.get().token_enum() == '=') {
      ReserveNewColumn(leaf, FlushLeft);
    }

    VLOG(2) << __FUNCTION__ << ", leaving leaf: " << leaf.get();
  }
};

// Distribution items should align on the :/ and := operators.
class DistItemColumnSchemaScanner : public VerilogColumnSchemaScanner {
 public:
  explicit DistItemColumnSchemaScanner(const FormatStyle &style)
      : VerilogColumnSchemaScanner(style) {}

 protected:
  void Visit(const SyntaxTreeNode &node) final {
    const auto tag = NodeEnum(node.Tag().tag);
    switch (tag) {
      case NodeEnum::kDistributionItem:
        // Start first column right away.
        item_column_ = ReserveNewColumn(node, FlushLeft);
        break;
      case NodeEnum::kValueRange: {
        if (!Context().DirectParentIs(NodeEnum::kDistributionItem)) {
          break;
        }
        CHECK_EQ(node.size(), 5);
        CHECK_NOTNULL(item_column_);
        ReserveNewColumn(item_column_, *node[0], FlushLeft,
                         GetSubpath(Path(), {0}));  // '['
        ReserveNewColumn(item_column_, *node[1], FlushRight,
                         GetSubpath(Path(), {1}));  // LHS value
        ReserveNewColumn(item_column_, *node[2], FlushLeft,
                         GetSubpath(Path(), {2}));  // ':'
        ReserveNewColumn(item_column_, *node[3], FlushRight,
                         GetSubpath(Path(), {3}));  // RHS value
        ReserveNewColumn(item_column_, *node[4], FlushLeft,
                         GetSubpath(Path(), {4}));  // ']'
        item_column_ = nullptr;
        return;
      }
      default:
        break;
    }

    TreeContextPathVisitor::Visit(node);  // Recurse down.
  }

  void Visit(const SyntaxTreeLeaf &leaf) final {
    switch (leaf.get().token_enum()) {
      case verilog_tokentype::TK_COLON_EQ:
      case verilog_tokentype::TK_COLON_DIV: {
        ReserveNewColumn(leaf, FlushLeft);
        break;
      }
      default:
        break;
    }
  }

 private:
  verible::ColumnPositionTree *item_column_ = nullptr;
};

static std::function<
    std::vector<TaggedTokenPartitionRange>(const TokenPartitionRange &)>
PartitionBetweenBlankLines(AlignableSyntaxSubtype subtype) {
  return [subtype](const TokenPartitionRange &range) {
    return verible::GetSubpartitionsBetweenBlankLinesSingleTag(
        range, static_cast<int>(subtype));
  };
}

// Each alignment group subtype maps to a set of functions.
struct AlignmentGroupHandlers {
  std::function<verible::AlignmentCellScannerFunction(
      const FormatStyle &vstyle)>
      column_scanner_func;
  std::function<verible::AlignmentPolicy(const FormatStyle &vstyle)>
      policy_func;
};

// Convert a pointer-to-member to a function/lambda that accesses that member.
// Returns the referenced member by value.
// TODO(fangism): move this to an STL-style util/functional library
template <typename MemberType, typename StructType>
static std::function<MemberType(const StructType &)>
function_from_pointer_to_member(MemberType StructType::*member) {
  return [member](const StructType &obj) { return obj.*member; };
}

using AlignmentHandlerMapType =
    std::map<AlignableSyntaxSubtype, AlignmentGroupHandlers>;

static void non_tree_column_scanner(
    verible::FormatTokenRange leading_tokens,
    verible::FormatTokenRange trailing_tokens,
    verible::ColumnPositionTree *column_entries) {
  static const SyntaxTreePath kLeadingTokensPath = {
      kLeadingNonTreeTokenPathIndex};
  static const SyntaxTreePath kTrailingCommaPath = {
      kTrailingNonTreeTokenPathIndex, 0};
  static const SyntaxTreePath kTrailingCommentPath = {
      kTrailingNonTreeTokenPathIndex, 1};

  VLOG(4) << __FUNCTION__ << "\nleading tokens: "
          << verible::StringSpanOfTokenRange(leading_tokens)
          << "\ntrailing tokens: "
          << verible::StringSpanOfTokenRange(trailing_tokens);

  if (!leading_tokens.empty()) {
    column_entries->Children().emplace_back(verible::ColumnPositionEntry{
        kLeadingTokensPath, *leading_tokens.front().token, FlushLeft});
  }

  if (trailing_tokens.empty()) return;

  const auto separator_it =
      std::find_if(trailing_tokens.begin(), trailing_tokens.end(),
                   [](const PreFormatToken &tok) {
                     return tok.TokenEnum() == ',' || tok.TokenEnum() == ':';
                   });

  auto comment_it = trailing_tokens.begin();

  if (separator_it != trailing_tokens.end()) {
    AlignmentColumnProperties prop;
    prop.contains_delimiter = true;
    const verible::ColumnPositionEntry column{kTrailingCommaPath,
                                              *separator_it->token, prop};
    column_entries->Children().emplace_back(column);

    comment_it = separator_it + 1;
  }
  if (comment_it != trailing_tokens.end() &&
      (comment_it->token->token_enum() == TK_COMMENT_BLOCK ||
       comment_it->token->token_enum() == TK_EOL_COMMENT)) {
    const verible::ColumnPositionEntry column{kTrailingCommentPath,
                                              *comment_it->token, FlushLeft};
    column_entries->Children().emplace_back(column);
  }
}

// Columns of an if/else-if/else branch row, for alignment like:
//   if (!rst_ni)     q <= '0;
//   else if (load_i) q <= d_i;
//   else             q <= q + 1;
// All branch rows are partitioned as [header] [statement], whereas their syntax
// tree origins differ, e.g. "else if" rows have a leading "else" outside of
// their origin. So the columns are taken from the partitions.
static verible::ColumnPositionTree ScanIfElseBranchColumns(
    const TokenPartitionTree &row) {
  verible::ColumnPositionTree columns({{}, verible::TokenInfo::EOFToken(), {}});
  const auto &parts = row.Children();
  if (parts.size() != 2) return columns;
  for (int i = 0; i < 2; ++i) {
    const auto tokens = parts[i].Value().TokensRange();
    if (tokens.empty()) return columns;
    columns.Children().emplace_back(verible::ColumnPositionEntry{
        SyntaxTreePath{i}, *tokens.front().token, FlushLeft});
  }
  return columns;
}

// Global registry of all known alignment handlers for Verilog.
// This organization lets the same handlers be re-used in multiple
// syntactic contexts, e.g. data declarations can be module items and
// generate items and block statement items.
static const AlignmentHandlerMapType &AlignmentHandlerLibrary() {
  static const auto *handler_map = new AlignmentHandlerMapType{
      {AlignableSyntaxSubtype::kDataDeclaration,
       {UnstyledAlignmentCellScannerGenerator<
            DataDeclarationColumnSchemaScanner>(),
        function_from_pointer_to_member(
            &FormatStyle::module_net_variable_alignment)}},
      {AlignableSyntaxSubtype::kNamedActualParameters,
       {UnstyledAlignmentCellScannerGenerator<
            ActualNamedParameterColumnSchemaScanner>(non_tree_column_scanner),
        function_from_pointer_to_member(
            &FormatStyle::named_parameter_alignment)}},
      {AlignableSyntaxSubtype::kNamedActualPorts,
       {UnstyledAlignmentCellScannerGenerator<
            ActualNamedPortColumnSchemaScanner>(non_tree_column_scanner),
        function_from_pointer_to_member(&FormatStyle::named_port_alignment)}},
      {AlignableSyntaxSubtype::kParameterDeclaration,
       {UnstyledAlignmentCellScannerGenerator<
            ParameterDeclarationColumnSchemaScanner>(non_tree_column_scanner),
        function_from_pointer_to_member(
            &FormatStyle::formal_parameters_alignment)}},
      {AlignableSyntaxSubtype::kBodyParameterDeclaration,
       {UnstyledAlignmentCellScannerGenerator<
            ParameterDeclarationColumnSchemaScanner>(non_tree_column_scanner),
        function_from_pointer_to_member(
            &FormatStyle::parameter_declaration_alignment)}},
      {AlignableSyntaxSubtype::kPortDeclaration,
       {UnstyledAlignmentCellScannerGenerator<
            PortDeclarationColumnSchemaScanner>(non_tree_column_scanner),
        function_from_pointer_to_member(
            &FormatStyle::port_declarations_alignment)}},
      {AlignableSyntaxSubtype::kStructUnionMember,
       {UnstyledAlignmentCellScannerGenerator<
            StructUnionMemberColumnSchemaScanner>(non_tree_column_scanner),
        function_from_pointer_to_member(
            &FormatStyle::struct_union_members_alignment)}},
      {AlignableSyntaxSubtype::kClassMemberVariables,
       {UnstyledAlignmentCellScannerGenerator<
            ClassPropertyColumnSchemaScanner>(),
        function_from_pointer_to_member(
            &FormatStyle::class_member_variable_alignment)}},
      {AlignableSyntaxSubtype::kCaseLikeItems,
       {UnstyledAlignmentCellScannerGenerator<CaseItemColumnSchemaScanner>(),
        function_from_pointer_to_member(&FormatStyle::case_items_alignment)}},
      {AlignableSyntaxSubtype::kContinuousAssignment,
       {UnstyledAlignmentCellScannerGenerator<AssignmentColumnSchemaScanner>(),
        function_from_pointer_to_member(
            &FormatStyle::assignment_statement_alignment)}},
      {AlignableSyntaxSubtype::kBlockingAssignment,
       {UnstyledAlignmentCellScannerGenerator<AssignmentColumnSchemaScanner>(),
        function_from_pointer_to_member(
            &FormatStyle::assignment_statement_alignment)}},
      {AlignableSyntaxSubtype::kNonBlockingAssignment,
       {UnstyledAlignmentCellScannerGenerator<AssignmentColumnSchemaScanner>(),
        function_from_pointer_to_member(
            &FormatStyle::assignment_statement_alignment)}},
      {AlignableSyntaxSubtype::kEnumListAssignment,
       {UnstyledAlignmentCellScannerGenerator<
            EnumWithAssignmentsColumnSchemaScanner>(non_tree_column_scanner),
        function_from_pointer_to_member(
            &FormatStyle::enum_assignment_statement_alignment)}},
      {AlignableSyntaxSubtype::kDistItem,
       {UnstyledAlignmentCellScannerGenerator<DistItemColumnSchemaScanner>(),
        function_from_pointer_to_member(
            &FormatStyle::distribution_items_alignment)}},
      {AlignableSyntaxSubtype::kNamedConstructorArguments,
       {UnstyledAlignmentCellScannerGenerator<
            NamedConstructorArgumentColumnSchemaScanner>(
            non_tree_column_scanner),
        function_from_pointer_to_member(&FormatStyle::named_port_alignment)}},
      {AlignableSyntaxSubtype::kTypedef,
       {UnstyledAlignmentCellScannerGenerator<TypedefColumnSchemaScanner>(
            non_tree_column_scanner),
        function_from_pointer_to_member(&FormatStyle::typedef_alignment)}},
      {AlignableSyntaxSubtype::kIfElseBranches,
       {[](const FormatStyle &) {
          return verible::AlignmentCellScannerFunction(
              &ScanIfElseBranchColumns);
        },
        function_from_pointer_to_member(
            &FormatStyle::if_else_statement_alignment)}},
  };
  return *handler_map;
}

static verible::AlignmentCellScannerFunction AlignmentColumnScannerSelector(
    const FormatStyle &vstyle, int subtype) {
  static const auto &handler_map = AlignmentHandlerLibrary();
  const auto iter = handler_map.find(AlignableSyntaxSubtype(subtype));
  CHECK(iter != handler_map.end()) << "subtype: " << subtype;
  return iter->second.column_scanner_func(vstyle);
}

static verible::AlignmentPolicy AlignmentPolicySelector(
    const FormatStyle &vstyle, int subtype) {
  static const auto &handler_map = AlignmentHandlerLibrary();
  const auto iter = handler_map.find(AlignableSyntaxSubtype(subtype));
  CHECK(iter != handler_map.end()) << "subtype: " << subtype;
  return iter->second.policy_func(vstyle);
}

static std::vector<AlignablePartitionGroup> ExtractAlignablePartitionGroups(
    const std::function<std::vector<TaggedTokenPartitionRange>(
        const TokenPartitionRange &)> &group_extractor,
    const verible::IgnoreAlignmentRowPredicate &ignore_group_predicate,
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  const std::vector<TaggedTokenPartitionRange> ranges(
      group_extractor(full_range));
  std::vector<AlignablePartitionGroup> groups;
  groups.reserve(ranges.size());
  for (const auto &range : ranges) {
    // Use the alignment scanner and policy that correspond to the
    // match_subtype.  This supports aligning a heterogenous collection of
    // alignable partition groups from the same parent partition (full_range).
    groups.emplace_back(
        FilterAlignablePartitions(range.range, ignore_group_predicate),
        AlignmentColumnScannerSelector(vstyle, range.match_subtype),
        AlignmentPolicySelector(vstyle, range.match_subtype));
    if (groups.back().IsEmpty()) groups.pop_back();
  }
  return groups;
}

using AlignSyntaxGroupsFunction =
    std::function<std::vector<AlignablePartitionGroup>(
        const TokenPartitionRange &range, const FormatStyle &style)>;

static std::vector<AlignablePartitionGroup> AlignPortDeclarations(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kPortDeclaration),
      &IgnoreWithinPortDeclarationPartitionGroup, full_range, vstyle);
}

static std::vector<AlignablePartitionGroup> AlignStructUnionMembers(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kStructUnionMember),
      &IgnoreWithinStructUnionMemberPartitionGroup, full_range, vstyle);
}

static std::vector<AlignablePartitionGroup> AlignActualNamedParameters(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(
          AlignableSyntaxSubtype::kNamedActualParameters),
      &IgnoreWithinActualNamedParameterPartitionGroup, full_range, vstyle);
}

static std::vector<AlignablePartitionGroup> AlignActualNamedPorts(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kNamedActualPorts),
      &IgnoreWithinActualNamedPortPartitionGroup, full_range, vstyle);
}

// Extracts alignable partition groups, optionally pre-splitting the range at
// blank lines when the alignment_group_boundary style setting requires it.
static std::vector<AlignablePartitionGroup>
ExtractAlignablePartitionGroupsWithBoundary(
    const std::function<std::vector<TaggedTokenPartitionRange>(
        const TokenPartitionRange &)> &group_extractor,
    const verible::IgnoreAlignmentRowPredicate &ignore_group_predicate,
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  if (!BlankLinesBreakGroups(vstyle.alignment_group_boundary)) {
    return ExtractAlignablePartitionGroups(
        group_extractor, ignore_group_predicate, full_range, vstyle);
  }
  // Pre-split at blank lines, then apply grouping to each sub-range.
  std::vector<AlignablePartitionGroup> all_groups;
  const auto sub_ranges =
      verible::GetSubpartitionsBetweenBlankLines(full_range);
  for (const auto &sub_range : sub_ranges) {
    auto groups = ExtractAlignablePartitionGroups(
        group_extractor, ignore_group_predicate, sub_range, vstyle);
    for (const auto &group : groups) {
      all_groups.push_back(group);
    }
  }
  return all_groups;
}

static std::vector<AlignablePartitionGroup> AlignModuleItems(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  // Applies to module/interface, generate, and package item lists.
  // Handles data/net/variable declarations, parameter/localparam
  // declarations, and continuous assignment statements.
  auto group_extractor = [&vstyle](const TokenPartitionRange &range) {
    return GetConsecutiveModuleItemGroups(range,
                                          vstyle.alignment_group_boundary);
  };
  return ExtractAlignablePartitionGroupsWithBoundary(
      group_extractor, &IgnoreCommentsAndPreprocessingDirectives, full_range,
      vstyle);
}

static std::vector<AlignablePartitionGroup> AlignClassItems(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  // TODO(fangism): align other class items besides member variables.
  auto group_extractor = [&vstyle](const TokenPartitionRange &range) {
    return GetConsecutiveClassItemGroups(
        range, vstyle.alignment_group_boundary,
        vstyle.align_class_parameter_declarations);
  };
  return ExtractAlignablePartitionGroupsWithBoundary(
      group_extractor, &IgnoreCommentsAndPreprocessingDirectives, full_range,
      vstyle);
}

static std::vector<AlignablePartitionGroup> AlignCaseItems(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kCaseLikeItems),
      &IgnoreMultilineCaseStatements, full_range, vstyle);
}

static std::vector<AlignablePartitionGroup> AlignEnumItems(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kEnumListAssignment),
      &IgnoreCommentsAndPreprocessingDirectives, full_range, vstyle);
}

// Aligns formal parameters in #(...) headers (module/interface/class port
// parameter lists).  Body-level parameter/localparam declarations are
// handled by AlignModuleItems which dispatches through
// GetConsecutiveModuleItemGroups.
static std::vector<AlignablePartitionGroup> AlignParameterDeclarations(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kParameterDeclaration),
      &IgnoreWithinPortDeclarationPartitionGroup, full_range, vstyle);
}

static std::vector<AlignablePartitionGroup> AlignStatements(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  auto group_extractor = [&vstyle](const TokenPartitionRange &range) {
    return GetAlignableStatementGroups(range, vstyle.alignment_group_boundary);
  };
  return ExtractAlignablePartitionGroupsWithBoundary(
      group_extractor, &IgnoreMultilineStatements, full_range, vstyle);
}

static std::vector<AlignablePartitionGroup> AlignDistItems(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kDistItem),
      &IgnoreCommentsAndPreprocessingDirectives, full_range, vstyle);
}

// Returns true for if/else branches that are not a single statement on one
// line, e.g. with a begin-end block or a comment forcing a line break, or
// that are not partitioned as [header] [statement].
static bool IsMultilineIfElseBranch(const TokenPartitionTree &partition) {
  // Like case items, the same tokens and comments make a row multi-line.
  return partition.Children().size() != 2 ||
         IgnoreMultilineCaseStatements(partition);
}

static std::vector<AlignablePartitionGroup> AlignIfElseBranches(
    const TokenPartitionRange &full_range, const FormatStyle &vstyle) {
  // Align either all branches of a chain or none of them: aligning some
  // branches around a begin-end block would look arbitrary.
  if (std::any_of(full_range.begin(), full_range.end(),
                  &IsMultilineIfElseBranch)) {
    return {};
  }
  return ExtractAlignablePartitionGroups(
      PartitionBetweenBlankLines(AlignableSyntaxSubtype::kIfElseBranches),
      &IsMultilineIfElseBranch, full_range, vstyle);
}

// Ignores comments and arguments other than ".name (value)".
static bool IgnoreWithinNamedConstructorArguments(
    const TokenPartitionTree &partition) {
  if (IgnoreCommentsAndPreprocessingDirectives(partition)) return true;
  const verible::Symbol *origin = partition.Value().Origin();
  return origin == nullptr ||
         origin->Tag() != verible::NodeTag(NodeEnum::kParamByName) ||
         !TokensHaveParenthesis(partition.Value().TokensRange());
}

void AlignNamedConstructorArguments(const FormatStyle &style,
                                    std::string_view full_text,
                                    const ByteOffsetSet &disabled_byte_ranges,
                                    TokenPartitionTree *partition_ptr) {
  const ExtractAlignmentGroupsFunction extract_alignment_groups =
      [&style](const TokenPartitionRange &full_range) {
        return ExtractAlignablePartitionGroups(
            PartitionBetweenBlankLines(
                AlignableSyntaxSubtype::kNamedConstructorArguments),
            &IgnoreWithinNamedConstructorArguments, full_range, style);
      };
  verible::TabularAlignTokens(style.column_limit, full_text,
                              disabled_byte_ranges, extract_alignment_groups,
                              partition_ptr);
}

// Returns true if the partition's rows are the branches of an if/else chain.
// This also covers chains that are the only statement of a block, whose
// partition is merged into the block's and keeps the block's origin.
static bool PartitionIsIfElseChain(const TokenPartitionTree &partition) {
  const auto &rows = partition.Children();
  if (rows.empty()) return false;
  const Symbol *first_row_origin = rows.front().Value().Origin();
  return first_row_origin != nullptr &&
         first_row_origin->Tag() == verible::NodeTag(NodeEnum::kIfClause);
}

void TabularAlignTokenPartitions(const FormatStyle &style,
                                 std::string_view full_text,
                                 const ByteOffsetSet &disabled_byte_ranges,
                                 TokenPartitionTree *partition_ptr) {
  VLOG(1) << __FUNCTION__;
  auto &partition = *partition_ptr;
  auto &uwline = partition.Value();
  const auto *origin = uwline.Origin();
  VLOG(2) << "origin is nullptr? " << (origin == nullptr);
  if (origin == nullptr) return;
  const SyntaxTreeNode *node = verible::MaybeNode(origin);
  VLOG(2) << "origin is node? " << (node != nullptr);
  if (node == nullptr) return;
  // Dispatch aligning function based on syntax tree node type.

  static const auto *const kAlignHandlers =
      new std::map<NodeEnum, AlignSyntaxGroupsFunction>{
          {NodeEnum::kPortDeclarationList, &AlignPortDeclarations},
          {NodeEnum::kPortList, &AlignPortDeclarations},
          {NodeEnum::kStructUnionMemberList, &AlignStructUnionMembers},
          {NodeEnum::kActualParameterByNameList, &AlignActualNamedParameters},
          {NodeEnum::kPortActualList, &AlignActualNamedPorts},
          // module/interface bodies
          {NodeEnum::kModuleItemList, &AlignModuleItems},
          {NodeEnum::kGenerateItemList, &AlignModuleItems},
          // package bodies
          {NodeEnum::kPackageItemList, &AlignModuleItems},
          {NodeEnum::kFormalParameterList, &AlignParameterDeclarations},
          {NodeEnum::kClassItems, &AlignClassItems},
          // various case-like constructs:
          {NodeEnum::kCaseItemList, &AlignCaseItems},
          {NodeEnum::kCaseInsideItemList, &AlignCaseItems},
          {NodeEnum::kGenerateCaseItemList, &AlignCaseItems},
          {NodeEnum::kEnumNameList, &AlignEnumItems},
          // align various statements, like assignments
          {NodeEnum::kStatementList, &AlignStatements},
          {NodeEnum::kBlockItemStatementList, &AlignStatements},
          {NodeEnum::kFunctionItemList, &AlignStatements},
          {NodeEnum::kDistributionItemList, &AlignDistItems},
      };
  AlignSyntaxGroupsFunction alignment_partitioner;
  if (PartitionIsIfElseChain(partition)) {
    alignment_partitioner = &AlignIfElseBranches;
  } else {
    const auto handler_iter = kAlignHandlers->find(NodeEnum(node->Tag().tag));
    if (handler_iter == kAlignHandlers->end()) return;
    alignment_partitioner = handler_iter->second;
  }

  const ExtractAlignmentGroupsFunction extract_alignment_groups =
      std::bind(alignment_partitioner, std::placeholders::_1, style);

  verible::TabularAlignTokens(style.column_limit, full_text,
                              disabled_byte_ranges, extract_alignment_groups,
                              &partition);

  VLOG(1) << "end of " << __FUNCTION__;
}

}  // namespace formatter
}  // namespace verilog
