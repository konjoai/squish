class Squish < Formula
  include Language::Python::Virtualenv

  desc "Local LLM server for Apple Silicon — paged KV cache, INT3 support"
  homepage "https://github.com/konjoai/squish"
  url "https://files.pythonhosted.org/packages/17/bb/d46eb909b6ff8c0430fe2e7e70f4ffde55df7b60acbfa856da24298a8347/squish_ai-9.33.4.tar.gz"
  sha256 "46e838a2777931d1f5373fa13183221be4159825ca65f93f9b841e7d7940b14f"
  bottle do
    root_url "https://github.com/konjoai/squish/releases/download/v9.33.4"
    rebuild 1
    sha256 cellar: :any, arm64_tahoe: "7a73c3e85b521477d0fe3e6e15e4367fc58b893fcfd5abf8b70dcca631843450"
  end
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

  def post_install
    system libexec/"bin/python3", "-c", "import squish"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/squish --version")
  end
end
