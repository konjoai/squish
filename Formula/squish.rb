class Squish < Formula
  include Language::Python::Virtualenv

  desc "Local LLM server for Apple Silicon — paged KV cache, INT3 support"
  homepage "https://github.com/konjoai/squish"
  url "https://files.pythonhosted.org/packages/46/f1/9dc04a4fc50e7a9b01b57fb340d1b99b98a426b0181fecd7e6667612ed25/squish_ai-9.32.0.tar.gz"
  sha256 "cdcbd32949b60caf7f30a44a3a2bd67598090c28376af4c4b6b303927ef41b02"
  license "BUSL-1.1"

  depends_on arch: :arm64
  depends_on :macos

  def python3
    which("python3.13") || which("python3.12") || which("python3.11") || which("python3.10") || which("python3")
  end

  def install
    py = python3
    virtualenv_create(libexec, py)
    system libexec/"bin/pip", "install", "--upgrade", "pip"
    system libexec/"bin/pip", "install", "squish-ai==#{version}"
    bin.install_symlink Dir["#{libexec}/bin/squish*"]
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/squish --version")
  end
end
