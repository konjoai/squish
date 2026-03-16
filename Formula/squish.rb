class Squish < Formula
  desc "Fast compressed model loader and OpenAI-compatible inference server for Apple Silicon"
  homepage "https://squishai.github.io/squish"
  url "https://github.com/squishai/squish/archive/refs/tags/v9.0.0.tar.gz"
  # sha256 — update after `gh release create v9.0.0` with:
  #   curl -sL https://github.com/squishai/squish/archive/refs/tags/v9.0.0.tar.gz | sha256sum
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/squishai/squish.git", branch: "main"

  livecheck do
    url :stable
    regex(/^v?(\d+(?:\.\d+)+)$/i)
  end

  depends_on "python@3.12"
  depends_on :macos
  depends_on arch: :arm64    # Apple Silicon (M1–M5) required

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install_and_link buildpath
    bin.install_symlink libexec/"bin/squish"
    bin.install_symlink libexec/"bin/squish-server"
    bin.install_symlink libexec/"bin/squish-convert"
  end

  def caveats
    <<~EOS
      Squish requires Apple Silicon (M1 or later) and macOS 13 Ventura+.
      Models are stored in ~/.squish/models/ by default.

      Get started (zero flags needed):
        squish run qwen3:8b

      Or run the interactive setup wizard:
        squish setup

      OpenAI-compatible API:
        curl http://localhost:11435/v1/chat/completions \\
          -H "Content-Type: application/json" \\
          -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Hello!"}]}'

      WhatsApp integration (Meta Cloud API):
        squish run qwen3:8b --whatsapp \\
          --whatsapp-verify-token  <token> \\
          --whatsapp-app-secret    <secret> \\
          --whatsapp-access-token  <token> \\
          --whatsapp-phone-number-id <id>
    EOS
  end

  test do
    assert_match "squish 9.0.0", shell_output("#{bin}/squish --version")
  end
end
