// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_UNIFORM_BIN_SLICER_H
#define XFIELDS_UNIFORM_BIN_SLICER_H


/*gpufun*/
void UniformBinSlicer_slice(UniformBinSlicerData el,
                LocalParticle* part0,
                int64_t const use_bunch_index_array,
                int64_t const use_slice_index_array,
                /*gpuglmem*/ int64_t* i_slice_part,
                /*gpuglmem*/ int64_t* i_bunch_part){

    int64_t const num_slices = UniformBinSlicerData_get_num_slices(el);
    double const z_min_edge = UniformBinSlicerData_get_z_min_edge(el);
    double const dzeta = UniformBinSlicerData_get_dzeta(el);

    int64_t const num_bunches = UniformBinSlicerData_get_num_bunches(el);
    double const bunch_spacing_zeta = UniformBinSlicerData_get_bunch_spacing_zeta(el);
    /*gpuglmem*/ double* num_particles = UniformBinSlicerData_getp1_num_particles(el, 0);

    // Generated by:
    // coords = ['x', 'px', 'y', 'py', 'zeta', 'delta']
    //     for cc in coords:
    //         print(f"double* sum_{cc} = (UniformBinSlicerData_len_sum_{cc}(el) > 0) ?  UniformBinSlicerData_getp1_sum_{cc}(el, 0) : NULL ;")
    /*gpuglmem*/ double* sum_x = (UniformBinSlicerData_len_sum_x(el) > 0) ?  UniformBinSlicerData_getp1_sum_x(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_px = (UniformBinSlicerData_len_sum_px(el) > 0) ?  UniformBinSlicerData_getp1_sum_px(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_y = (UniformBinSlicerData_len_sum_y(el) > 0) ?  UniformBinSlicerData_getp1_sum_y(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_py = (UniformBinSlicerData_len_sum_py(el) > 0) ?  UniformBinSlicerData_getp1_sum_py(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_zeta = (UniformBinSlicerData_len_sum_zeta(el) > 0) ?  UniformBinSlicerData_getp1_sum_zeta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_delta = (UniformBinSlicerData_len_sum_delta(el) > 0) ?  UniformBinSlicerData_getp1_sum_delta(el, 0) : NULL ;

    // Generated by:
    // moms = ['x_x', 'x_px', 'x_y', 'x_py', 'x_zeta', 'x_delta',
    //         'px_px', 'px_y', 'px_py', 'px_zeta', 'px_delta',
    //         'y_y', 'y_py', 'y_zeta', 'y_delta',
    //         'py_py', 'py_zeta', 'py_delta',
    //         'zeta_zeta', 'zeta_delta',
    //         'delta_delta']
    // for mm in moms:
    //    print(f"double* sum_{mm} = (UniformBinSlicerData_len_sum_{mm}(el)) ? UniformBinSlicerData_getp1_sum_{mm}(el, 0) : NULL ;")
    /*gpuglmem*/ double* sum_x_x = (UniformBinSlicerData_len_sum_x_x(el)) ? UniformBinSlicerData_getp1_sum_x_x(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_x_px = (UniformBinSlicerData_len_sum_x_px(el)) ? UniformBinSlicerData_getp1_sum_x_px(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_x_y = (UniformBinSlicerData_len_sum_x_y(el)) ? UniformBinSlicerData_getp1_sum_x_y(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_x_py = (UniformBinSlicerData_len_sum_x_py(el)) ? UniformBinSlicerData_getp1_sum_x_py(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_x_zeta = (UniformBinSlicerData_len_sum_x_zeta(el)) ? UniformBinSlicerData_getp1_sum_x_zeta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_x_delta = (UniformBinSlicerData_len_sum_x_delta(el)) ? UniformBinSlicerData_getp1_sum_x_delta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_px_px = (UniformBinSlicerData_len_sum_px_px(el)) ? UniformBinSlicerData_getp1_sum_px_px(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_px_y = (UniformBinSlicerData_len_sum_px_y(el)) ? UniformBinSlicerData_getp1_sum_px_y(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_px_py = (UniformBinSlicerData_len_sum_px_py(el)) ? UniformBinSlicerData_getp1_sum_px_py(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_px_zeta = (UniformBinSlicerData_len_sum_px_zeta(el)) ? UniformBinSlicerData_getp1_sum_px_zeta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_px_delta = (UniformBinSlicerData_len_sum_px_delta(el)) ? UniformBinSlicerData_getp1_sum_px_delta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_y_y = (UniformBinSlicerData_len_sum_y_y(el)) ? UniformBinSlicerData_getp1_sum_y_y(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_y_py = (UniformBinSlicerData_len_sum_y_py(el)) ? UniformBinSlicerData_getp1_sum_y_py(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_y_zeta = (UniformBinSlicerData_len_sum_y_zeta(el)) ? UniformBinSlicerData_getp1_sum_y_zeta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_y_delta = (UniformBinSlicerData_len_sum_y_delta(el)) ? UniformBinSlicerData_getp1_sum_y_delta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_py_py = (UniformBinSlicerData_len_sum_py_py(el)) ? UniformBinSlicerData_getp1_sum_py_py(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_py_zeta = (UniformBinSlicerData_len_sum_py_zeta(el)) ? UniformBinSlicerData_getp1_sum_py_zeta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_py_delta = (UniformBinSlicerData_len_sum_py_delta(el)) ? UniformBinSlicerData_getp1_sum_py_delta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_zeta_zeta = (UniformBinSlicerData_len_sum_zeta_zeta(el)) ? UniformBinSlicerData_getp1_sum_zeta_zeta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_zeta_delta = (UniformBinSlicerData_len_sum_zeta_delta(el)) ? UniformBinSlicerData_getp1_sum_zeta_delta(el, 0) : NULL ;
    /*gpuglmem*/ double* sum_delta_delta = (UniformBinSlicerData_len_sum_delta_delta(el)) ? UniformBinSlicerData_getp1_sum_delta_delta(el, 0) : NULL ;

    uint8_t needs_cov = sum_x_px || sum_x_y || sum_x_py || sum_x_zeta || sum_x_delta ||
                        sum_px_y || sum_px_py || sum_px_zeta || sum_px_delta ||
                        sum_y_py || sum_y_zeta || sum_y_delta ||
                        sum_py_zeta || sum_py_delta ||
                        sum_zeta_delta;
    uint8_t needs_x_plane = sum_x || sum_x_x || sum_px || sum_px_px;
    uint8_t needs_y_plane = sum_y || sum_y_y || sum_py || sum_py_py;
    uint8_t needs_z_plane = sum_zeta || sum_zeta_zeta || sum_delta || sum_delta_delta;

    //start_per_particle_block (part0->part)
        double zeta = LocalParticle_get_zeta(part);
        double weight = LocalParticle_get_weight(part);
        const int64_t ipart = part->ipart;

        int64_t i_bunch_in_slicer = -1;
        uint8_t bunch_is_in_slicer = 0;
        int64_t this_i_slot_part;


        // identify i_bunch_in_slicer and absolute slot number
        if (num_bunches == 0){
            i_bunch_in_slicer = 0;
            bunch_is_in_slicer = 1;
            this_i_slot_part = 0;
        }
        else {
            this_i_slot_part = -floor((zeta - z_min_edge) / bunch_spacing_zeta);
            for(int ii = 0; ii < num_bunches; ii++){
                if(UniformBinSlicerData_get_filled_slots(el,
                    UniformBinSlicerData_get_bunch_numbers(el,ii)) == this_i_slot_part) {
                    bunch_is_in_slicer = 1;
                    i_bunch_in_slicer = ii;
                    break;
                }
            }
        }

        // Write to memory i_bunch_part
        if (use_bunch_index_array){
            if (bunch_is_in_slicer == 1){
                i_bunch_part[ipart] = i_bunch_in_slicer;
            } else {
                i_bunch_part[ipart] = -1;
            }
        }

        // handle the slices
        if (bunch_is_in_slicer != 1){
            if (use_slice_index_array){
                i_slice_part[ipart] = -1;
            }
        } else {
            double z_min_edge_bunch;
            if (num_bunches <= 1){
                z_min_edge_bunch = z_min_edge;
            }
            else{
                z_min_edge_bunch = z_min_edge - this_i_slot_part * bunch_spacing_zeta;
            }

            int64_t i_slice = floor((zeta - z_min_edge_bunch) / dzeta);
            if (i_slice >= 0 && i_slice < num_slices){
                if (use_slice_index_array){
                    i_slice_part[ipart] = i_slice;
                }

                atomicAdd(&num_particles[i_slice + i_bunch_in_slicer * num_slices], weight);

                // Generated by the following and then reshuffled:
                // coords = ['x', 'px', 'y', 'py', 'zeta', 'delta']
                // for cc in coords:
                //     print(f"if (sum_{cc}) atomicAdd(&sum_{cc}[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_{cc}(part));")
                // moms = ['x_x', 'px_px', 'y_y', 'py_py', 'zeta_zeta', 'delta_delta']
                // for mm in moms:
                //     cc1, cc2 = mm.split('_')
                //     print(f"if (sum_{mm})  atomicAdd(&sum_{mm}[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_{cc1}(part) * LocalParticle_get_{cc2}(part));")
                if (needs_x_plane){
                    if (sum_x) atomicAdd(&sum_x[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_x(part));
                    if (sum_x_x)  atomicAdd(&sum_x_x[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_x(part) * LocalParticle_get_x(part));
                    if (sum_px) atomicAdd(&sum_px[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_px(part));
                    if (sum_px_px)  atomicAdd(&sum_px_px[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_px(part) * LocalParticle_get_px(part));
                }
                if (needs_y_plane){
                    if (sum_y) atomicAdd(&sum_y[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_y(part));
                    if (sum_y_y)  atomicAdd(&sum_y_y[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_y(part) * LocalParticle_get_y(part));
                    if (sum_py) atomicAdd(&sum_py[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_py(part));
                    if (sum_py_py)  atomicAdd(&sum_py_py[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_py(part) * LocalParticle_get_py(part));
                }
                if (needs_z_plane){
                    if (sum_zeta) atomicAdd(&sum_zeta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_zeta(part));
                    if (sum_zeta_zeta)  atomicAdd(&sum_zeta_zeta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_zeta(part) * LocalParticle_get_zeta(part));
                    if (sum_delta) atomicAdd(&sum_delta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_delta(part));
                    if (sum_delta_delta)  atomicAdd(&sum_delta_delta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_delta(part) * LocalParticle_get_delta(part));
                }
                // Generated by:
                // moms = ['x_px', 'x_y', 'x_py', 'x_zeta', 'x_delta',
                //         'px_y', 'px_py', 'px_zeta', 'px_delta',
                //         'y_py', 'y_zeta', 'y_delta',
                //         'py_zeta', 'py_delta',
                //         'zeta_delta']
                // for mm in moms:
                //     cc1, cc2 = mm.split('_')
                //     print(f"if (sum_{mm})  atomicAdd(&sum_{mm}[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_{cc1}(part) * LocalParticle_get_{cc2}(part));")
                if (needs_cov){
                    if (sum_x_px)  atomicAdd(&sum_x_px[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_x(part) * LocalParticle_get_px(part));
                    if (sum_x_y)  atomicAdd(&sum_x_y[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_x(part) * LocalParticle_get_y(part));
                    if (sum_x_py)  atomicAdd(&sum_x_py[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_x(part) * LocalParticle_get_py(part));
                    if (sum_x_zeta)  atomicAdd(&sum_x_zeta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_x(part) * LocalParticle_get_zeta(part));
                    if (sum_x_delta)  atomicAdd(&sum_x_delta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_x(part) * LocalParticle_get_delta(part));
                    if (sum_px_y)  atomicAdd(&sum_px_y[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_px(part) * LocalParticle_get_y(part));
                    if (sum_px_py)  atomicAdd(&sum_px_py[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_px(part) * LocalParticle_get_py(part));
                    if (sum_px_zeta)  atomicAdd(&sum_px_zeta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_px(part) * LocalParticle_get_zeta(part));
                    if (sum_px_delta)  atomicAdd(&sum_px_delta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_px(part) * LocalParticle_get_delta(part));
                    if (sum_y_py)  atomicAdd(&sum_y_py[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_y(part) * LocalParticle_get_py(part));
                    if (sum_y_zeta)  atomicAdd(&sum_y_zeta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_y(part) * LocalParticle_get_zeta(part));
                    if (sum_y_delta)  atomicAdd(&sum_y_delta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_y(part) * LocalParticle_get_delta(part));
                    if (sum_py_zeta)  atomicAdd(&sum_py_zeta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_py(part) * LocalParticle_get_zeta(part));
                    if (sum_py_delta)  atomicAdd(&sum_py_delta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_py(part) * LocalParticle_get_delta(part));
                    if (sum_zeta_delta)  atomicAdd(&sum_zeta_delta[i_slice + i_bunch_in_slicer * num_slices], weight * LocalParticle_get_zeta(part) * LocalParticle_get_delta(part));
                }
            } else{
                if (use_slice_index_array){
                    i_slice_part[ipart] = -1;
                }
            }
        }


    //end_per_particle_block

}

// Dummy, not to upset the metaclass

/*gpufun*/
void UniformBinSlicer_track_local_particle(UniformBinSlicerData el,
                LocalParticle* part0){

}


#endif
