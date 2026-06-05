import SwiftUI

// ── Markdown rendering for chat messages ──────────────────────────────────
//
// SwiftUI's built-in `AttributedString(markdown:)` handles inline formatting
// (bold, italic, code, links) but NOT block structure — headings, lists, code
// fences. This view parses block structure manually and delegates inline
// formatting to AttributedString per block, which gives us:
//
//   • # / ## / ### headings
//   • Numbered (1. 2. 3.) and bullet (- *) lists with indent
//   • ``` fenced code blocks
//   • Bold **x** / italic *x* / inline code `x` inside paragraphs and list items
//
// Re-parsing on every streamed token is fine — markdown buffers stay short and
// the parser is O(n).

struct MarkdownView: View {
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(parseMarkdownBlocks(content)) { block in
                blockView(block)
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: MdBlock) -> some View {
        switch block.kind {
        case .heading(let level):
            inlineText(block.text)
                .font(.system(size: headingSize(level), weight: .bold))
                .foregroundStyle(SQ.textPrimary)
                .padding(.top, level == 1 ? 6 : 2)
        case .listItem(let marker, let indent):
            HStack(alignment: .top, spacing: 8) {
                Text(marker)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(SQ.accentBright)
                    .frame(minWidth: 14, alignment: .trailing)
                inlineText(block.text)
                    .font(.system(size: 14))
                    .foregroundStyle(SQ.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, CGFloat(indent) * 16)
        case .paragraph:
            inlineText(block.text)
                .font(.system(size: 14))
                .foregroundStyle(SQ.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
        case .codeBlock(let language):
            VStack(alignment: .leading, spacing: 4) {
                if !language.isEmpty {
                    Text(language)
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .foregroundStyle(SQ.textSecond)
                        .kerning(0.5)
                }
                Text(block.text)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(SQ.textPrimary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .padding(10)
            .background(SQ.surface2)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(SQ.border, lineWidth: 1)
            )
            .cornerRadius(6)
        }
    }

    /// Render inline markdown (bold/italic/code/links) within a single block.
    private func inlineText(_ s: String) -> Text {
        var options = AttributedString.MarkdownParsingOptions()
        options.interpretedSyntax = .inlineOnlyPreservingWhitespace
        if let attr = try? AttributedString(markdown: s, options: options) {
            return Text(attr)
        }
        return Text(s)
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: return 20
        case 2: return 17
        case 3: return 15
        default: return 14
        }
    }
}

// ── Block model ────────────────────────────────────────────────────────────

struct MdBlock: Identifiable {
    enum Kind: Equatable {
        case heading(Int)
        case listItem(marker: String, indent: Int)
        case paragraph
        case codeBlock(language: String)
    }
    let id = UUID()
    let kind: Kind
    let text: String
}

// ── Parser ─────────────────────────────────────────────────────────────────

private let numberedListRE: NSRegularExpression = {
    // Captures: 1=digit run, 2=item body
    // swiftlint:disable:next force_try
    try! NSRegularExpression(pattern: #"^(\d+)\.\s+(.+)$"#)
}()

func parseMarkdownBlocks(_ content: String) -> [MdBlock] {
    var blocks: [MdBlock] = []
    var inCode = false
    var codeLang = ""
    var codeBuffer: [String] = []
    var paraBuffer: [String] = []

    func flushParagraph() {
        guard !paraBuffer.isEmpty else { return }
        let joined = paraBuffer.joined(separator: " ")
        blocks.append(MdBlock(kind: .paragraph, text: joined))
        paraBuffer.removeAll()
    }
    func flushCode() {
        let joined = codeBuffer.joined(separator: "\n")
        blocks.append(MdBlock(kind: .codeBlock(language: codeLang), text: joined))
        codeBuffer.removeAll()
        codeLang = ""
    }

    for rawLine in content.components(separatedBy: "\n") {
        let line = rawLine
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        // ── Code fence open/close ────────────────────────────────────────
        if trimmed.hasPrefix("```") {
            if inCode {
                flushCode()
                inCode = false
            } else {
                flushParagraph()
                inCode = true
                codeLang = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            }
            continue
        }
        if inCode {
            codeBuffer.append(line)
            continue
        }

        // ── Heading ───────────────────────────────────────────────────────
        if let h = matchHeading(trimmed) {
            flushParagraph()
            blocks.append(MdBlock(kind: .heading(h.level), text: h.text))
            continue
        }

        // ── List item ─────────────────────────────────────────────────────
        if let item = matchListItem(line) {
            flushParagraph()
            blocks.append(MdBlock(
                kind: .listItem(marker: item.marker, indent: item.indent),
                text: item.text
            ))
            continue
        }

        // ── Blank line → paragraph break ─────────────────────────────────
        if trimmed.isEmpty {
            flushParagraph()
            continue
        }

        // ── Otherwise: accumulate into current paragraph ─────────────────
        paraBuffer.append(trimmed)
    }

    if inCode { flushCode() }
    flushParagraph()
    return blocks
}

private func matchHeading(_ s: String) -> (level: Int, text: String)? {
    if s.hasPrefix("### ") { return (3, String(s.dropFirst(4))) }
    if s.hasPrefix("## ")  { return (2, String(s.dropFirst(3))) }
    if s.hasPrefix("# ")   { return (1, String(s.dropFirst(2))) }
    return nil
}

private func matchListItem(_ raw: String) -> (marker: String, indent: Int, text: String)? {
    // Count leading spaces / tabs for indent level — 2 spaces = 1 indent.
    var indentChars = 0
    var idx = raw.startIndex
    while idx < raw.endIndex, raw[idx] == " " || raw[idx] == "\t" {
        indentChars += raw[idx] == "\t" ? 4 : 1
        idx = raw.index(after: idx)
    }
    let body = String(raw[idx...])
    let indent = indentChars / 2

    // Numbered: "1. xxx"
    let nsRange = NSRange(body.startIndex..., in: body)
    if let m = numberedListRE.firstMatch(in: body, range: nsRange),
       let r1 = Range(m.range(at: 1), in: body),
       let r2 = Range(m.range(at: 2), in: body) {
        return ("\(body[r1]).", indent, String(body[r2]))
    }
    // Bullets: "- xxx" or "* xxx" (require space after marker so we don't
    // mistake "*emphasis*" for a list).
    if body.hasPrefix("- ") { return ("•", indent, String(body.dropFirst(2))) }
    if body.hasPrefix("* ") { return ("•", indent, String(body.dropFirst(2))) }
    return nil
}
