class Squish < Formula
  include Language::Python::Virtualenv

  desc "Local LLM server for Apple Silicon — paged KV cache, INT3 support"
  homepage "https://github.com/konjoai/squish"
  url "https://files.pythonhosted.org/packages/80/82/71a12b87ddc4ca00d5a650427c11a0c2e12beb61518cfb55a3ec79c94843/squish_ai-9.33.5.tar.gz"
  sha256 "7e399c3155bd16b03cc62d4774b60688ef31d18f473e0cd3338cb9c0a3a5b9ec"
  bottle do
    root_url "https://github.com/konjoai/squish/releases/download/v9.33.5"
    sha256 cellar: :any_skip_relocation, arm64_tahoe: "7d3b0f5cac08e178f6635a5581e02c7632882331b32d26c23c7ce13bad327937"
  end
  license "BUSL-1.1"

  depends_on "python@3.13"
  depends_on arch: :arm64
  depends_on :macos

  def install
    py = Formula["python@3.13"].opt_bin/"python3.13"
    virtualenv_create(libexec, py)
    system libexec/"bin/python3", "-m", "ensurepip"
    system libexec/"bin/pip", "install",
           "--no-warn-script-location",
           "squish-ai==#{version}"
    bin.install_symlink Dir["#{libexec}/bin/squish*"]
  end

  def post_install
    system libexec/"bin/python3", "-c", "import squish"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/squish --version")
  end
end
