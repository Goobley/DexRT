Features/ToDo
=============

- [x] Basic radiance cascade method brought from python impl (@22299f7)
- [x] Update raymarcher to Amanatides & Woo formalism (@fcdf129)
- [x] Reformulate for flatland participating media (@c808b28)
- [x] Convert to 2.5D (@c808b28)
- [x] Track down issues on top edge (@95994d0)
- [x] Compare with Lightweaver on simple case (FS only) (@7beb0fb)
- [x] Add mipmapping for upper levels (@5f65d5c)
- [x] Add branching/bilinear fix and compare (@5f65d5c)
- [x] CRTAF parsing (@ed49231)
- [x] Load atmosphere (@85ed408)
- [x] LTE populations (@b54f9ff)
- [x] Line absorption profile (@c878288)
- [x] Emission, opacity, source function (@2bf9af7 - not source function, but that's just a ratio!)
- [x] LTE formal solution from atmosphere (@bbddad4 - static only)
- [ ] LTE background from tabulated atmosphere
- [ ] LTE with background J iteration
- [x] Final formal solution (interactive?)
    - Mostly working. EmisOpac system needs to be refactored to handle "random" wavelengths
- [x] Non-LTE iteration (@5a68d12)
    - [x] Add line profile normalisation in final cascade (`wphi`)
- [ ] Boundary conditions
    - [x] Interpolated Prom style
        - Full 3D treatment available (but only considering ray projection in
        x-z), applying a correction factor for solid angle due to muz
    - [ ] Periodic
        - Lagged sampling from previous cascade 0? -- very diffusive
        - Ping-pong full cascade stack and do a directional lookup/traversal
            - Could do this in a separate alg and build a BC from this
        - Hauschildt & Baron style multiple wrap-around?
    - [x] Make generic via templating struct passed down to raymarcher -- each BC
    has a `sample` function so dispatch is fully static (there won't be many
    types so use a switch to template dispatch). Needs some tidying to wrap up exploding cases as BCs are added
- [x] Multiple atoms - basically splat everything SOA a la MULTI and like is done for the one atom here. -- Messy, but done
- [x] Charge conservation
    - [ ] Finite difference charge conservation blows up if we're off the end of the grid -- makes sense; dC will also explode
- [x] Pressure conservation
- [x] Config file/input flags
- [ ] Pydantic objects for input files
    - [x] Dexrt
    - [x] Ray
- [x] Active atmosphere cells (temperature criterion)
- [x] Active probes (from C0 active cells)
- [x] Save compiled params into output
    - [ ] Add attributes to netcdf layer
        - [x] Not done properly, but key attributes added by shoving through netcdf-c functions directly.
- [x] Control dexrt output
- [ ] Allow loading populations into "lte" mode.
- [x] Avoid writing nonsense from non-active probes in C0 into J when pingponging
    - We will get this for free when the sparsity propagates there.
- [ ] Add groups to netcdf layer
- [ ] Save/Restart from snapshot
- [x] Embed git hash in build (https://jonathanhamberg.com/post/cmake-embedding-git-hash/)
- [x] Sparse VDB-like grid
    - Welcome MrBlockMap 💅
- [x] Optionally page J out to host memory (enabled by default)
- [x] Add method to IndexGenerators to generate the flat array index equivalent, we've messed that up enough.
    - This is now available as .full_flat_index()
- [x] Finish migrating Classic emis/opac method to MRBlockMap
    - [ ] Remove old DDA infrastructure (after dexrt_ray updated)
        - Not if we leave `dexrt_ray` as-is
        - Removed `dda_raymarch_2d` but left old `RayMarchState2d` infrastructure.
    - [x] Propagate `mip_chain` allocation into main -- can store in CascadeState
- [ ] Update ProbesToCompute to launch in normal block order
- [x] Output sparse data by default, but have bool to rehydrate before saving (and support doing so in dexrt_py)
    - [x] Output `max_mip_level` for each wave_batch
    - [x] Output active map: sufficient information to reconstruct the tiles and their locations from flat buffers. probably just block_map.active_tiles that we can morton decode.
    - [x] Add extra attrs for sparse config (e.g. BLOCK_SIZE)
- [ ] Load sparse output into `dexrt_ray`
    - [ ] Just rehydrate the atmosphere and leave as-is
    - [ ] Will need to create own `mr_block_map` but with `max_mip_level` 0 from data in output.
- [x] Add extra attrs for emis/opac config
- [ ] Support ANGLE_INVARIANT_THERMAL_VEL_FRAC for CoreAndVoigt?
- [ ] Make dexrt_py a proper package
    - [ ] Get onto pypi
    - [ ] Add tonemapping code
    - [ ] Handle sparse rehydration
- [x] Only allocate necessary Gamma and pops (all driven by BlockMap)
    - [ ] Preallocate and store LTE pops... they're allocated 99% of the time currently.
        - Can still update every iteration (essentially free)
- [ ] Set mip variance limits in config file.
    - Parsing logic now in
- [x] Set mip levels in config file.
- [x] Set max cascade in config file.
- [x] Create sparse atmos and only keep that one on device
    - Migrate everything to the sparse atmos (that aligns with the active probes)
- [x] Move more things into .cpps to improve compile time.
    - Didn't have the largest effect, but many incremental compiles are faster
- [ ] Pull out user config variables in Config header into something less busy
- [ ] Optimise accumulation into Gamma
- [ ] Support for Golding method
- [ ] PRD
    - ML ?
        - Jrest from J and v
        - directly to rho
    - Paletou 1995 method?


Ideas
=====

- [ ] Move LTE populations to be partition function based (allows for calculation of only the levels of interest)
- [x] Use magma to abstract batched LU solved for populations
    - [ ] Needs non-GPU alternative
- [x] Store rays per cascade pre-averaged (groups of 4) - should reduce bandwidth and storage requirements, but requires atomic operation (available in refactor)
- [x] Avoid local memory in dda ray traversal -- seems to be causing stalls
    - Implemented fully register-based approach using the switch/template method from nanovdb - big perf difference.
- [x] Full per-wavelength active set treatment. GPU benefits a lot more from this
- [x] Bring back wavelength batches -- consider a warp (32 threads) of inclination rays (e.g. 4) with e.g. 8 wavelengths. The raymarching will be entirely coherent for these. Emissivity/Opacity gather will be almost perfectly coherent too. In 3D, if we have memory do full warps of wavelengths to get this coherence back.
- [x] Refactor to only have one raymarch/RC impl
- [x] Handle case of solving one direction of c0 at a time (with all necessary components of upper cascades) - same memory as preaveraging, but useful for e.g. dynamic models in 3D
- [x] Is it possible to create a basis of emissivity and opacity that can be interpolated as a function of mux/muy/muz to allow emissivity/opacity to be computed for fewer directions, mipmapped, and then linearly combined in-situ? Needs tests
    - This is done on a velocity-dependent basis in DirectionalEmisOpacInterp... remains quite memory intensive for low error
    - Other mipmappable option is CoreAndVoigt --  we store the line core parameters (eta*, chi*, a_damp, inv_dop_width), and modulate them with the Voigt.
- [ ] Sparse line quadratures that fit entirely inside a WAVE_BATCH (can be increased). Ensure whole line is done in one go, them use a higher order scheme to evaluate the wavelength integral over I?
- [ ] If we stick with such a simple ALO, it can actually be computed in-situ when computing Gamma, saving the memory.


Notes
=====

Formalising coordinate system
------------------------------
- z up, x default perpendicular axis, y as per right-hand rule
- Storage is as [z, x] or [z, y, x]
- These correspond to probes v, u in 2D, and w, v, u in 3D.
- Indexing as wave typically implies the member of a batch, and la the index into the wavelength array
- ks for sparse spatial index
- tile_idx is the index of the contiguous cartesian block, block_idx is the index _within_ the block.


Weird Issues
============

[x] Some array accesses, such as the BlockMap entries from MultiLevelDDA are
triggering the YAKL "host array being accessed in a device kernel issues", but
compute-sanitizer does not report an issue. This problem also occured when
copying the mips out to 2d arrays for debugging.
    - This wasn't really a false positive, although the description was wrong.
    mr_block_map was accessing block_map via a pointer, which was to the host
    device, but the cuda compiler was optimising and pulling the device pointer
    in block_map through, which was the only bit needed _unless_ the debug mode
    was activated...

[ ] Magma 2.8.0 builds on nvhpc 24.7 (not on 24.9) but does not yield correct results. Current git HEAD works on 24.7