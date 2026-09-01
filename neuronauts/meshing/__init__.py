"""Mesh project results for Neuroglancer and other 3D tools.

Everything this project produces with a shape -- v117 atoms with their real L2
adjacency and caliber, kimimaro/CAVE skeletons, ``Fragment`` collections,
assembled neurons -- becomes a :class:`SkeletonGeometry`, is swept into a tube
mesh (:func:`tube_mesh`), and is written as a bundle (:func:`export_bundle`)
that Neuroglancer loads as a segmentation layer with meshes and skeletons.
Assembly results ride along as ``equivalences`` so an assembled neuron is one
colour without re-meshing.

    from neuronauts.meshing import SkeletonGeometry, export_bundle
    skel = SkeletonGeometry(vertices_nm, edges, radii_nm)
    export_bundle("viz/mesh/demo", {864691135361314119: skel})
    # python scripts/mesh_results.py serve --dir viz/mesh  -> open viz/mesh/demo/url.txt

Only numpy is required. ``cloud-volume`` is used by the tests to cross-check
the precomputed bytes; the ``neuroglancer`` package is optional for a local
viewer (:func:`launch_local_viewer`).
"""

from neuronauts.meshing.bundle import (
    DEFAULT_BASE_URL, MeshParams, build_state, export_bundle, load_manifest, write_state,
)
from neuronauts.meshing.formats import (
    encode_precomputed_mesh, encode_precomputed_skeleton, write_obj, write_ply,
    write_precomputed_mesh_dir, write_precomputed_skeleton_dir, write_segment_properties,
)
from neuronauts.meshing.serve import launch_local_viewer, serve_forever, serve_in_thread
from neuronauts.meshing.skeleton import SkeletonGeometry, concat_skeletons
from neuronauts.meshing.tube import TriMesh, skeleton_chains, tube_mesh

__all__ = [
    "DEFAULT_BASE_URL", "MeshParams", "SkeletonGeometry", "TriMesh",
    "build_state", "concat_skeletons", "encode_precomputed_mesh",
    "encode_precomputed_skeleton", "export_bundle", "launch_local_viewer",
    "load_manifest", "serve_forever", "serve_in_thread", "skeleton_chains",
    "tube_mesh", "write_obj", "write_ply", "write_precomputed_mesh_dir",
    "write_precomputed_skeleton_dir", "write_segment_properties", "write_state",
]
