from cmaketools import setup

setup(
    name="TFC",
    version="0.0.1",
    author="Carl Leake",
    author_email="leakec57@gmail.com",
    description="A test package for TFC.",
    url="https://github.com/leakec/tfc.git",
    license="None",
    src_dir="src",
    has_package_data=False,
    install_requires=["cmaketools","numpy","jax","jaxlib"]
)
