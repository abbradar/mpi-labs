let
  config = {
  };

  pkgs = import <nixpkgs> { inherit config; };

in pkgs.clangStdenv.mkDerivation {
  name = "mpi-lab";

  buildInputs = with pkgs; [ openmpi ];

  nativeBuildInputs = with pkgs; [ pkgconfig cmake ];
}
