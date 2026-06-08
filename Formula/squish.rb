class Squish < Formula
  include Language::Python::Virtualenv

  desc "Local LLM server for Apple Silicon — paged KV cache, INT3 support"
  homepage "https://github.com/konjoai/squish"
  url "https://files.pythonhosted.org/packages/48/ea/2a464510e954adf65724412d75817b40e94483cbfcda6fa7992ca6a01eb4/squish_ai-9.33.7.tar.gz"
  sha256 "5ee3bef38e5041e31f76fa3812e27a36fc5b8803436908cd0f20fa992f453f5b"
  license "BUSL-1.1"

  resource "annotated-doc" do
    url "https://files.pythonhosted.org/packages/1e/d3/26bf1008eb3d2daa8ef4cacc7f3bfdc11818d111f7e2d0201bc6e3b49d45/annotated_doc-0.0.4-py3-none-any.whl"
    sha256 "571ac1dc6991c450b25a9c2d84a3705e2ae7a53467b5d111c24fa8baabbed320"
  end

  resource "annotated-types" do
    url "https://files.pythonhosted.org/packages/78/b6/6307fbef88d9b5ee7421e68d78a9f162e0da4900bc5f5793f6d3d0e34fb8/annotated_types-0.7.0-py3-none-any.whl"
    sha256 "1f02e8b43a8fbbc3f3e0d4f0f4bfc8131bcb4eebe8849b8e5c773f3a1c582a53"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/da/42/e921fccf5015463e32a3cf6ee7f980a6ed0f395ceeaa45060b61d86486c2/anyio-4.13.0-py3-none-any.whl"
    sha256 "08b310f9e24a9594186fd75b4f73f4a4152069e3853f1ed8bfbf58369f4ad708"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/59/8c/57e832b7af6d7c5abe66eb3fbe3a3a32f4d11ea23a1aa7131371035be991/certifi-2026.5.20-py3-none-any.whl"
    sha256 "3c52e209ba0a4ad7aebe60436a4ab349c39e1e602e8c134221e546902ad25897"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/c7/0d/67e5b4109ea4a837e80daa87c2c696711955e40449a97e8926672534def2/click-8.4.1-py3-none-any.whl"
    sha256 "482be17c6991b8c19c5429a1e995d9b0efdbb63172824c41f99965dc0ade8ec2"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/e0/82/45359b62a067409bd929ae8a56b8ed13e5a8c8a61194b3c236920999ab83/fastapi-0.136.3-py3-none-any.whl"
    sha256 "3d2a69bdf04b7e9f3afa292c3bc7a98816bbfafa10bc9b45f3f3700d2f761620"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/d5/0c/043d5e551459da400957a1395e0febbf771446ff34291afcbe3d8be2a279/fsspec-2026.4.0-py3-none-any.whl"
    sha256 "11ef7bb35dab8a394fde6e608221d5cf3e8499401c249bebaeaad760a1a8dec2"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl"
    sha256 "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"
  end

  resource "hf-xet" do
    url "https://files.pythonhosted.org/packages/9b/ff/edcc2b40162bef3ff78e14ab637e5f3b89243d6aee72f5949d3bb6a5af83/hf_xet-1.5.0-cp37-abi3-macosx_11_0_arm64.whl"
    sha256 "fd6e5a9b0fdac4ed03ed45ef79254a655b1aaab514a02202617fbf643f5fdf7a"
  end

  resource "httpcore" do
    url "https://files.pythonhosted.org/packages/7e/f5/f66802a942d491edb555dd61e3a9961140fd64c90bce1eafd741609d334d/httpcore-1.0.9-py3-none-any.whl"
    sha256 "2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/2a/39/e50c7c3a983047577ee07d2a9e53faf5a69493943ec3f6a384bdc792deb2/httpx-0.28.1-py3-none-any.whl"
    sha256 "d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad"
  end

  resource "huggingface_hub" do
    url "https://files.pythonhosted.org/packages/0b/03/40a05316cb6616e5b7efd7773656441ab04b4b022c2199e79bb4622a92a3/huggingface_hub-1.18.0-py3-none-any.whl"
    sha256 "729be4a976fb706dcc02d176bcda8a3f32bdf21a294e8f4b3dda6fbcbc9c1ab1"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/1e/5e/d4e9f1a599fb8e573b7b87160658329fbf28d19eac2718f51fc3def3aa5a/idna-3.18-py3-none-any.whl"
    sha256 "7f952cbe720b688055e3f87de14f5c3e5fdaa8bc3928985c4077ca689de849a2"
  end

  resource "Jinja2" do
    url "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl"
    sha256 "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"
  end

  resource "markdown-it-py" do
    url "https://files.pythonhosted.org/packages/b3/81/4da04ced5a082363ecfa159c010d200ecbd959ae410c10c0264a38cac0f5/markdown_it_py-4.2.0-py3-none-any.whl"
    sha256 "9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a"
  end

  resource "MarkupSafe" do
    url "https://files.pythonhosted.org/packages/9c/d9/5f7756922cdd676869eca1c4e3c0cd0df60ed30199ffd775e319089cb3ed/markupsafe-3.0.3-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "116bb52f642a37c115f517494ea5feb03889e04df47eeff5b130b1808ce7c219"
  end

  resource "mdurl" do
    url "https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl"
    sha256 "84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8"
  end

  resource "mlx" do
    url "https://files.pythonhosted.org/packages/a3/3f/888f8664d4f8e23a1363a5f50024be5216e199ab7ad0ba20988c7ed6d729/mlx-0.31.2-cp313-cp313-macosx_14_0_arm64.whl"
    sha256 "1b3fb0dda955b0d552ce57bdd6f42b3309ab21b067e40587d6848443d307e91f"
  end

  resource "mlx-lm" do
    url "https://files.pythonhosted.org/packages/90/02/9a67b8e4f87e3e2e5cd7b1ad79304b93c09a0db6af34bee75e6551c06c60/mlx_lm-0.31.3-py3-none-any.whl"
    sha256 "758cfddf1180053b7613db76fad3d246a331a2a905808e1164a275621fc983b8"
  end

  resource "mlx-metal" do
    url "https://files.pythonhosted.org/packages/3f/69/fe3b783ebe999f3118234e1e940feb622518bfb1dea6ac5d13b1d36a8449/mlx_metal-0.31.2-py3-none-macosx_14_0_arm64.whl"
    sha256 "b25385bcee18fc194092255b8b53b9a3d8489eb650e59160f1b57aadd07aa2dc"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/1b/30/a80189bcc7f5e4258b3fbc3968d909d1756f54d023299ecc39ad6fdb9ef8/numpy-2.4.6-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "bf162abab1c1a736333192707cef898e735a5ca00f38f27eeedf44b39d9e85eb"
  end

  resource "orjson" do
    url "https://files.pythonhosted.org/packages/bc/f7/7118f965541aeac6844fcb18d6988e111ac0d349c9b80cda53583e758908/orjson-3.10.18-cp313-cp313-macosx_15_0_arm64.whl"
    sha256 "1ebeda919725f9dbdb269f59bc94f861afbe2a27dce5608cdba2d92772364d1c"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/df/b2/87e62e8c3e2f4b32e5fe99e0b86d576da1312593b39f47d8ceef365e95ed/packaging-26.2-py3-none-any.whl"
    sha256 "5fc45236b9446107ff2415ce77c807cee2862cb6fac22b8a73826d0693b0980e"
  end

  resource "protobuf" do
    url "https://files.pythonhosted.org/packages/b8/ef/50433d346c56657a70d27f156c7b349ac59a068b01de4eb796e747eecc43/protobuf-7.35.0-py3-none-any.whl"
    sha256 "c13f325cf242bad135c350629eeb5d54b24228eb472fb3e2e9ebbd4c5dc20ca0"
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/fd/7b/122376b1fd3c62c1ed9dc80c931ace4844b3c55407b6fb2d199377c9736f/pydantic-2.13.4-py3-none-any.whl"
    sha256 "45a282cde31d808236fd7ea9d919b128653c8b38b393d1c4ab335c62924d9aba"
  end

  resource "pydantic_core" do
    url "https://files.pythonhosted.org/packages/c1/81/4fa520eaffa8bd7d1525e644cd6d39e7d60b1592bc5b516693c7340b50f1/pydantic_core-2.46.4-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "c94f0688e7b8d0a67abf40e57a7eaaecd17cc9586706a31b76c031f63df052b4"
  end

  resource "Pygments" do
    url "https://files.pythonhosted.org/packages/f4/7e/a72dd26f3b0f4f2bf1dd8923c85f7ceb43172af56d63c7383eb62b332364/pygments-2.20.0-py3-none-any.whl"
    sha256 "81a9e26dd42fd28a23a2d169d86d7ac03b46e2f8b59ed4698fb4785f946d0176"
  end

  resource "PyYAML" do
    url "https://files.pythonhosted.org/packages/b1/16/95309993f1d3748cd644e02e38b75d50cbc0d9561d21f390a76242ce073f/pyyaml-6.0.3-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "2283a07e2c21a2aa78d9c4442724ec1eb15f5e42a723b99cb3d822d48f5f7ad1"
  end

  resource "regex" do
    url "https://files.pythonhosted.org/packages/2d/e7/d0eaf5713828417b9e5648cf81fa9bacd4961f6ab98c380c2034f8716e35/regex-2026.5.9-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "a8820737949116ffff55fe18f9fc644530063ba6ebfcb8314239416e78f1347c"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/82/3b/64d4899d73f91ba49a8c18a8ff3f0ea8f1c1d75481760df8c68ef5235bf5/rich-15.0.0-py3-none-any.whl"
    sha256 "33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb"
  end

  resource "safetensors" do
    url "https://files.pythonhosted.org/packages/e8/00/374c0c068e30cd31f1e1b46b4b5738168ec79e7689ca82ee93ddfea05109/safetensors-0.7.0-cp38-abi3-macosx_11_0_arm64.whl"
    sha256 "94fd4858284736bb67a897a41608b5b0c2496c9bdb3bf2af1fa3409127f20d57"
  end

  resource "sentencepiece" do
    url "https://files.pythonhosted.org/packages/8d/de/5a007fb53b1ab0aafc69d11a5a3dd72a289d5a3e78dcf2c3a3d9b14ffe93/sentencepiece-0.2.1-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "097f3394e99456e9e4efba1737c3749d7e23563dd1588ce71a3d007f25475fff"
  end

  resource "shellingham" do
    url "https://files.pythonhosted.org/packages/e0/f9/0595336914c5619e5f28a1fb793285925a8cd4b432c9da0a987836c7f822/shellingham-1.5.4-py2.py3-none-any.whl"
    sha256 "7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686"
  end

  resource "sse-starlette" do
    url "https://files.pythonhosted.org/packages/dc/67/805710444ea8cc75fbf70b920ed431a560c4bf9c57f7d5a3117213189399/sse_starlette-3.4.4-py3-none-any.whl"
    sha256 "3f4dd50d8aed2771a091f3a83000323fc3844541c16b4fe585ae2420cc6df973"
  end

  resource "starlette" do
    url "https://files.pythonhosted.org/packages/1c/54/196d0c1db10af76baa4f64894448505d60d3cdf70ef92cbb35f46a4e4c71/starlette-1.2.1-py3-none-any.whl"
    sha256 "4de0082d08c8f6764a85a54cf1120d6939507a19905c7768acad2a9f875d2b89"
  end

  resource "tokenizers" do
    url "https://files.pythonhosted.org/packages/2e/47/174dca0502ef88b28f1c9e06b73ce33500eedfac7a7692108aec220464e7/tokenizers-0.22.2-cp39-abi3-macosx_11_0_arm64.whl"
    sha256 "1e418a55456beedca4621dbab65a318981467a2b188e982a23e117f115ce5001"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/47/aa/218a0eb34de1f753c83e4d0d1c8e7c4cef27f20dcb8342e024f63a80dc86/tqdm-4.68.1-py3-none-any.whl"
    sha256 "fea4a90e4023f764914569f7802a297277c5ab1a66be5144143e142e1a4031d8"
  end

  resource "transformers" do
    url "https://files.pythonhosted.org/packages/73/6f/e1564b0cc182afa05e219a8e09a8e770ffaab879b6b824b56c819bd221da/transformers-5.10.2-py3-none-any.whl"
    sha256 "8a669db546f82c7c3618cb46ceb0f0afd89292bc70f319c058f8332ec63e268d"
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/3f/f9/2b3ff4e56e5fa7debfaf9eb135d0da96f3e9a1d5b27222223c7296336e5f/typer-0.25.1-py3-none-any.whl"
    sha256 "75caa44ed46a03fb2dab8808753ffacdbfea88495e74c85a28c5eefcf5f39c89"
  end

  resource "typing_extensions" do
    url "https://files.pythonhosted.org/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl"
    sha256 "f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548"
  end

  resource "typing-inspection" do
    url "https://files.pythonhosted.org/packages/dc/9b/47798a6c91d8bdb567fe2698fe81e0c6b7cb7ef4d13da4114b41d239f65d/typing_inspection-0.4.2-py3-none-any.whl"
    sha256 "4ed1cacbdc298c220f1bd249ed5287caa16f34d44ef4e9c3d0cbad5b521545e7"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/88/fa/e1388bbcf24ef3274f45c0c1c7b501fd14971037c1b6ee23610553307497/uvicorn-0.49.0-py3-none-any.whl"
    sha256 "ba3d14c3ee7e41c6c654c46c9eb489d33213cdd30aa1696eab1374337c13f68f"
  end

  resource "zstandard" do
    url "https://files.pythonhosted.org/packages/3f/06/9ae96a3e5dcfd119377ba33d4c42a7d89da1efabd5cb3e366b156c45ff4d/zstandard-0.25.0-cp313-cp313-macosx_11_0_arm64.whl"
    sha256 "a1a4ae2dec3993a32247995bdfe367fc3266da832d82f8438c8570f989753de1"
  end

  depends_on "python@3.13"
  depends_on arch: :arm64
  depends_on :macos

  # Install every resource directly from its cached download path. Brew's
  # cache filenames are prefixed with a content hash (e.g.
  # "abc123--foo-1.0.whl"), which pip rejects as an invalid wheel name, so
  # we copy each file to a clean basename in a scratch dir first. This
  # bypasses Homebrew's virtualenv_install_with_resources, which extracts
  # wheels into directories pip can't install and which passes
  # --no-binary :all: to pip — blocking even build dependencies from
  # using wheels.
  def install
    virtualenv_create(libexec, Formula["python@3.13"].opt_bin/"python3.13")

    scratch = buildpath/"deps"
    scratch.mkpath

    install_from_cache = lambda do |cached|
      clean = scratch/cached.basename.to_s.sub(/^[0-9a-f]+--/, "")
      FileUtils.cp(cached, clean)
      system libexec/"bin/pip", "install", "--no-deps",
             "--no-warn-script-location", clean
    end

    resources.each do |r|
      r.fetch
      install_from_cache.call(r.cached_download)
    end

    system libexec/"bin/pip", "install", "--no-deps",
           "--no-warn-script-location", buildpath

    bin.install_symlink Dir["#{libexec}/bin/squish*"]
  end

  def post_install
    system libexec/"bin/python3", "-c", "import squish"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/squish --version")
  end
end
