class Squish < Formula
  include Language::Python::Virtualenv

  desc "Local LLM server for Apple Silicon — paged KV cache, INT3 support"
  homepage "https://github.com/konjoai/squish"
  url "https://files.pythonhosted.org/packages/c9/f2/31a4274e633d73f67838a7ee8561857c5bb72e746e4d75b45c3d3fb11dd8/squish_ai-9.33.2.tar.gz"
  sha256 "e189f9c042455bd35b8b8078ab18e25f1f449578ae9843a159e3fc30af3c5280"
  license "BUSL-1.1"

  depends_on "python@3.13"
  depends_on arch: :arm64
  depends_on :macos

  def install
    # Explicitly use brew Python, never system Python
    py = Formula["python@3.13"].opt_bin/"python3.13"
    virtualenv_create(libexec, py)
    system libexec/"bin/pip", "install", "--upgrade", "pip"
    system libexec/"bin/pip", "install", "squish-ai==#{version}"
    bin.install_symlink Dir["#{libexec}/bin/squish*"]
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/squish --version")
  end
end
