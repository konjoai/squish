import SwiftUI

enum SQ {
    static let bg          = Color(hex:"#09071a")
    static let surface     = Color(hex:"#100d24")
    static let surface2    = Color(hex:"#16112e")
    static let border      = Color(hex:"#2a1d4a")
    static let accent      = Color(hex:"#8B5CF6")
    static let accentBright = Color(hex:"#A78BFA")
    static let accentPink  = Color(hex:"#EC4899")
    static let textPrimary = Color(hex:"#e8e0f8")
    static let textSecond  = Color(hex:"#7c6ea8")
    static let green       = Color(hex:"#4ade80")
    static let red         = Color(hex:"#f87171")
    static let amber       = Color(hex:"#fbbf24")
    static let userBubble  = Color(hex:"#3b1d8a")
    static let asstBubble  = Color(hex:"#13102a")
}

extension Color {
    init(hex: String) {
        let h = hex.trimmingCharacters(in: .init(charactersIn:"#"))
        var rgb: UInt64 = 0
        Scanner(string: h).scanHexInt64(&rgb)
        self.init(
            red:   Double((rgb >> 16) & 0xFF) / 255,
            green: Double((rgb >>  8) & 0xFF) / 255,
            blue:  Double( rgb        & 0xFF) / 255
        )
    }
}
