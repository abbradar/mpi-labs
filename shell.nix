let
  config = {
  };

  pkgs = import <nixpkgs> { inherit config; };

in pkgs.clangStdenv.mkDerivation {
  name = "mpi-lab";

  buildInputs = with pkgs; [ openmpi ];

  nativeBuildInputs = with pkgs; [ pkgconfig cmake ];

  OMPI_CC = "clang";
  OMPI_CXX = "clang++";
}
