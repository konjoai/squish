class Squish < Formula
  include Language::Python::Virtualenv

  desc "Local LLM server for Apple Silicon — paged KV cache, INT3 support"
  homepage "https://github.com/konjoai/squish"
  url "https://files.pythonhosted.org/packages/4f/e5/ee45f0dc181afbcc43541adb068a210aac92be9451e5ae32fba6e6cfb9a5/squish_ai-9.33.1.tar.gz"
  sha256 "e12efd7351c7b7e726b00ed7230a58820a32b4877b203c154ace68ee767d928c"
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
