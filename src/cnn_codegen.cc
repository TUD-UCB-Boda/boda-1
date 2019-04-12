// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"

#include"rtc_func_gen.H"
#include"geom_prim.H"

namespace boda
{
  struct cnn_custom_codegen_t : public custom_codegen_t {

    virtual void gen_op( rtc_call_gen_t * rcg, string const & op_name ) {
      // *** custom codegen hooks ***
      if( op_name == "conv" ) { gen_op_conv(rcg); }
      else if( op_name == "conv_simd" ) { gen_op_conv_simd(rcg); }
      else if( op_name == "ipconv" ) { gen_op_ipconv(rcg); }
      else if( op_name == "k1conv" ) { gen_op_k1conv(rcg); }
      else if( op_name == "k1conv_simd" ) { gen_op_k1conv_simd(rcg); }
      else if( op_name == "tconv" ) { gen_op_tconv(rcg); }
      else if( op_name == "sgemm" ) { gen_op_sgemm(rcg); }
      else if( op_name == "sgemm_no_local" ) { gen_op_sgemm_no_local(rcg); }
      else if( op_name == "sgemm_simd" ) { gen_op_sgemm_simd(rcg); }
      else if( op_name == "sgemm_simd_local" ) { gen_op_sgemm_simd_local(rcg); }
      else if( op_name == "bconv" ) { gen_op_bconv(rcg); }
      else if( op_name == "bconv_fb" ) { gen_op_bconv_fb(rcg); }
      else if( op_name == "reduce" ) { gen_op_reduce(rcg); }
      else if( op_name == "wgconv" ) { gen_op_wgconv(rcg); }
      else if( op_name == "wgconv_xpose_filts") { gen_op_wgconv_filts(rcg);}
      else if( op_name == "wgconv_xpose_in") { gen_op_wgconv_in(rcg);}
      else if( op_name == "wgconv_xpose_out") { gen_op_wgconv_out(rcg);}
      else if( op_name == "wfconv") { gen_op_wfconv(rcg);}
    }

    void gen_op_wfconv( rtc_call_gen_t * rcg ) {
      //dims_t const & in_ref = rcg->get_arg_dims_by_name("out_ref");
      dims_t const & work = rcg->get_arg_dims_by_name("work");
      dims_t const & img = rcg->get_arg_dims_by_name("in");
      dims_t const & filts = rcg->get_arg_dims_by_name("filts");

      uint32_t r = filts.dsz("x");
      uint32_t a = work.dsz("a");
      uint32_t o = a - r + 1;
      uint32_t out_tile = o;

      printf("RA: %i, %i, %i", r, a, o);
      rcg->set("w_A", str(a));
      rcg->set("halfAA", str(a*a/2));
      rcg->set("w_O", str(o));
      rcg->set("w_R", str(r));
      rcg->set("w_rSize", str(a*2));

      uint32_t TPmask, TPwidth, TPshift, TQmask, TQwidth, TQshift, Nmask, Nwidth;
      uint32_t N = img.dsz("img");
      if (N < 2)
        TPmask = 0x18, TPwidth = 2, TPshift = 3, TQmask = 0x07, TQwidth = 3, TQshift = 0, Nmask = 0x00, Nwidth = 0;
      else if (N < 4)
        TPmask = 0x18, TPwidth = 2, TPshift = 3, TQmask = 0x06, TQwidth = 2, TQshift = 1, Nmask = 0x01, Nwidth = 1;
      else if (N < 8)
        TPmask = 0x10, TPwidth = 1, TPshift = 4, TQmask = 0x0c, TQwidth = 2, TQshift = 2, Nmask = 0x03, Nwidth = 2;
      else if (N < 16)
        TPmask = 0x10, TPwidth = 1, TPshift = 4, TQmask = 0x08, TQwidth = 1, TQshift = 3, Nmask = 0x07, Nwidth = 3;
      else if (N < 32)
        TPmask = 0x00, TPwidth = 0, TPshift = 0, TQmask = 0x10, TQwidth = 1, TQshift = 4, Nmask = 0x0f, Nwidth = 4;
      else
        TPmask = 0x00, TPwidth = 0, TPshift = 0, TQmask = 0x00, TQwidth = 0, TQshift = 0, Nmask = 0x1f, Nwidth = 5;


      rcg->set("win_TP", str( u32_ceil_div(img.dsz("x"), 2 << TPwidth)));
      rcg->set("win_TQ", str( u32_ceil_div(img.dsz("y"), 2 << TQwidth)));

      string const simple_superblocking = strprintf(
          "    int TPmask = %d, TPwidth = %d, TPshift = %d,"
          "        TQmask = %d, TQwidth = %d, TQshift = %d,"
          "        Nmask = %d, Nwidth = %d;"
          , TPmask, TPwidth, TPshift, TQmask, TQwidth, TQshift, Nmask, Nwidth);

      rcg->set("superblocking", simple_superblocking);
      rcg->set("win_TK", str( u32_ceil_div(rcg->get_arg_dims_by_name("out").dsz("chan"), 32)));
      rcg->set("win_TN", str( u32_ceil_div(img.dsz("img"), 1 << Nwidth)));

      if (tc == nullptr) {
        tc = new tc_gen(o,r);
        /*tc->gen_points();
        tc->gen_code();*/
        tc->gen_code_py(work.dsz("ul"));

      }
      rcg->set("trans_i1", tc->code_Bta);
      rcg->set("trans_i2", tc->code_Btb);
      rcg->set("trans_f1", tc->code_Ga);
      rcg->set("trans_f2", tc->code_Gb);
      rcg->set("trans_o1", tc->code_Ata);
      rcg->set("trans_o2", tc->code_Atb);

    }

    void gen_op_wgconv( rtc_call_gen_t * rcg ) {
      dims_t const & in_ref = rcg->get_arg_dims_by_name("in_ref");
      dims_t const & work = rcg->get_arg_dims_by_name("work");

      uint32_t r = work.dsz("r");
      uint32_t a = work.dsz("a");
      uint32_t o = a - r + 1;
      rcg->set("w_P", str(u32_ceil_div(in_ref.dsz("x"), o)));
      rcg->set("w_Q", str(u32_ceil_div(in_ref.dsz("y"), o))); // TODO
    }

    void gen_op_wgconv_out( rtc_call_gen_t * rcg ) {
      dims_t const & in_ref = rcg->get_arg_dims_by_name("out_ref");
      dims_t const & work = rcg->get_arg_dims_by_name("work");

      uint32_t r = work.dsz("r");
      uint32_t a = work.dsz("a");
      uint32_t o = a - r + 1;

      printf("RA: %i, %i, %i", r, a, o);
      rcg->set("w_A", str(a));
      rcg->set("w_O", str(o));
      rcg->set("w_R", str(r));
      rcg->set("w_P", str(u32_ceil_div(in_ref.dsz("x"), o))); // TODO
      rcg->set("w_Q", str(u32_ceil_div(in_ref.dsz("y"), o))); // TODO
      rcg->set("w_W", str(in_ref.dsz("y")));
      rcg->set("w_H", str(in_ref.dsz("x")));
      //rcg->set("w_C", str(in_ref.dsz("chan")));
      rcg->set("w_N", str(in_ref.dsz("img")));
      rcg->set("w_K", str(in_ref.dsz("chan")));

      if (tc == nullptr) { // redudant here but just to be sure
        tc = new tc_gen(o,r);
        //tc->gen_points();
        tc->gen_code_py(work.dsz("ul"));
      }
      rcg->set("first_mul", tc->code_Ata);
      rcg->set("second_mul", tc->code_Atb);

    }

    void gen_op_wgconv_in( rtc_call_gen_t * rcg ) {
      dims_t const & in_ref = rcg->get_arg_dims_by_name("in_ref");
      dims_t const & work = rcg->get_arg_dims_by_name("work");

      uint32_t r = work.dsz("r");
      uint32_t a = work.dsz("a");
      uint32_t o = a - r + 1;

      // For now thread block size is tba x tba.
      uint32_t tba = 32;
      uint32_t shared_mem = a * (tba) - (tba-2)*(r-1);
      rcg->set("w_shared_image", str(shared_mem*shared_mem));
      rcg->set("w_instead", str(tba*tba*a*a));
      rcg->set("w_load_thread", str(shared_mem*shared_mem/(tba*tba)));

      rcg->set("w_A", str(a));
      rcg->set("w_O", str(o));
      rcg->set("w_R", str(r));
      rcg->set("w_P", str(u32_ceil_div(in_ref.dsz("y"), o))); // TODO
      rcg->set("w_Q", str(u32_ceil_div(in_ref.dsz("x"), o))); // TODO
      rcg->set("w_W", str(in_ref.dsz("y")));
      rcg->set("w_H", str(in_ref.dsz("x")));
      rcg->set("w_C", str(in_ref.dsz("chan")));
      rcg->set("w_N", str(in_ref.dsz("img")));

      printf("RA: %i, %i, %i", r, a, o);
      if (tc == nullptr) { // redudant here but just to be sure
        tc = new tc_gen(o,r);
        //tc->gen_points();
        tc->gen_code_py(work.dsz("ul"));
      }
      rcg->set("first_mul", tc->code_Bta);
      rcg->set("second_mul", tc->code_Btb);
    }
    void gen_op_wgconv_filts( rtc_call_gen_t * rcg ) {
      // implement me
      dims_t const & filts = rcg->get_arg_dims_by_name("filts_ref");
      dims_t const & work = rcg->get_arg_dims_by_name("work");

      uint32_t r = filts.dsz("x");
      uint32_t a = work.dsz("a");
      uint32_t o = a - r + 1;

      rcg->set("w_A", str(a));
      rcg->set("w_R", str(r));
      rcg->set("w_C", str(filts.dsz("in_chan")));
      rcg->set("w_K", str(filts.dsz("out_chan")));

      printf("RA: %i, %i", r, a);
      if (tc == nullptr) {
        tc = new tc_gen(o,r);
        //tc->gen_points();
        //tc->gen_code();
        tc->gen_code_py(work.dsz("ul"));
      }
      rcg->set("first_mul", tc->code_Ga);
      rcg->set("second_mul", tc->code_Gb);
    }

    void gen_op_reduce( rtc_call_gen_t * rcg ) {
      // FIXME: iterator over non-flat args and/or select ins arg directly, then do inner iter?
      for( vect_arg_decl_t::multi_iter i = rcg->rtc_func_template->arg_decls.multi_begin( &rcg->op ); !i.at_end(); ++i ) {
        if( i.ad().vn != "ins" ) { continue; }
	rcg->line( "ins_ops", "v += "+i.vn()+"[GLOB_ID_1D];" );
      }
    }
    string maybe_add_relu( rtc_call_gen_t * rcg, string const & ve ) {
      return rcg->op.get_u32("conv_has_relu") ? ( "max(0.0f,"+ve+")" ) : ve;
    }

    string add_bias_then_maybe_relu( rtc_call_gen_t * rcg, dims_t const & work, uint32_t const & tx, uint32_t const ty ) {
      string const ve = strprintf( "(out_tile[%s] + filts_strip[%s])", str((ty*work.dsz("out_chan")+tx)).c_str(), str(tx).c_str() );
      return maybe_add_relu( rcg, ve );
    }

    void gen_op_bconv( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      uint32_t const in_smem_sz = work.dsz("pels_tile")*work.dsz("pels");
      rcg->set( "in_smem_sz", str(in_smem_sz) );
      uint32_t const in_smem_load_iter = u32_ceil_div( in_smem_sz, rcg->rtc_call_geom.tpb );
      rcg->set( "in_smem_load_iter", str(in_smem_load_iter) );

      uint32_t const filts_smem_sz = work.dsz("out_ix_tile")*work.dsz("out_ix");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const filts_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->rtc_call_geom.tpb );
      rcg->set( "filts_smem_load_iter", str(filts_smem_load_iter) );

      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem[%%(LOC_ID_1D_out_ix_tile)*%%(work_out_ix_dim)+%s];",
					 str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem[%%(LOC_ID_1D_pels_tile)*%%(work_pels_dim)+%s];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }

      rcg->line( "outs_to_filts_strip", "switch(work_pel) { " );
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "outs_to_filts_strip", "case "+str(ty)+":" );
	for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	  uint32_t const rix = ty*work.dsz("out_ix")+tx;
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];",
					  str(rix).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "outs_to_filts_strip", strprintf( "filts_strip[%s] = out_tile[%s];",
					    str(tx).c_str(), str(rix).c_str() ) );
	}
	rcg->line( "outs_to_filts_strip", "break;" );
      }
      rcg->line( "outs_to_filts_strip", "} " );

      string store_expr = R"foo(
  igl_y = (%(pel_ix_y)-%(bck_in_pad_y_dim))*%(stride_y_dim)+%(out_ix_sy)-%(in_pad_y_dim)+%(bck_pad_in_off_y_dim);
  igl_x = (%(pel_ix_x)-%(bck_in_pad_x_dim))*%(stride_x_dim)+%(out_ix_sx)-%(in_pad_x_dim)+%(bck_pad_in_off_x_dim);
  if( igl_x >= 0 && igl_y >= 0 && igl_y < %(in_grad_loss_y_dim) && igl_x < %(in_grad_loss_x_dim) &&
      %(out_ix_in_chan) < %(in_grad_loss_chan_dim) && %(pel_ix_img) < %(in_grad_loss_img_dim) ) {
    in_grad_loss[ %(pel_ix_img)*%(in_grad_loss_img_stride) + %(out_ix_in_chan)*%(in_grad_loss_chan_stride) +
		  igl_y*%(in_grad_loss_y_stride) + igl_x*%(in_grad_loss_x_stride)] = filts_strip[)foo";
      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "stores", store_expr + strprintf( "%s];\n};", str(tx).c_str() ) );
	rcg->line( "stores", "++out_ix;" );
      }
    }

    void gen_op_bconv_fb( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work_fb" );
      uint32_t const in_smem_sz = work.dsz("pels_tile")*work.dsz("pels");
      rcg->set( "in_smem_sz", str(in_smem_sz) );
      uint32_t const in_smem_load_iter = u32_ceil_div( in_smem_sz, rcg->rtc_call_geom.tpb );
      rcg->set( "in_smem_load_iter", str(in_smem_load_iter) );

      uint32_t const filts_smem_sz = work.dsz("out_ix_tile")*work.dsz("out_ix");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const filts_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->rtc_call_geom.tpb );
      rcg->set( "filts_smem_load_iter", str(filts_smem_load_iter) );

      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem[%%(LOC_ID_1D_out_ix_tile)*%%(work_fb_out_ix_dim)+%s];",
					 str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem[%%(LOC_ID_1D_pels_tile)*%%(work_fb_pels_dim)+%s];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }

      rcg->line( "outs_to_filts_strip", "switch(work_pel) { " );
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "outs_to_filts_strip", "case "+str(ty)+":" );
	for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	  uint32_t const rix = ty*work.dsz("out_ix")+tx;
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];",
					  str(rix).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "outs_to_filts_strip", strprintf( "filts_strip[%s] = out_tile[%s];",
					    str(tx).c_str(), str(rix).c_str() ) );
	}
	rcg->line( "outs_to_filts_strip", "break;" );
      }
      rcg->line( "outs_to_filts_strip", "} " );

      string store_expr = R"foo(
  if( %(pel_ix_in_chan) < %(filts_grad_loss_in_chan_dim) && %(out_ix_out_chan) < %(filts_grad_loss_out_chan_dim) ) {
    filts_grad_loss[ %(out_ix_out_chan)*%(filts_grad_loss_out_chan_stride) + %(pel_ix_in_chan)*%(filts_grad_loss_in_chan_stride) +
		  %(pel_ix_y)*%(filts_grad_loss_y_stride) + %(pel_ix_x)*%(filts_grad_loss_x_stride)] = filts_strip[)foo";
      for( uint32_t tx = 0; tx != work.dsz( "out_ix" ); ++tx ) {
	rcg->line( "stores", store_expr + strprintf( "%s];\n};", str(tx).c_str() ) );
	rcg->line( "stores", "++out_ix;" );
      }
    }

    void gen_filts_smem_loads( rtc_call_gen_t * rcg, uint32_t const filts_smem_sz ) { // note: filts_smem_sz must == tvv %(filts_smem_sz)
      uint32_t const out_chan_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->rtc_call_geom.tpb );
      for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rcg->rtc_call_geom.tpb > filts_smem_sz ) {
	  rcg->line( "filts_smem_loads", "if( "+ixe+" < %(filts_smem_sz) ) {" );eif = "}";}
	// note: load is (always) contiguous
	rcg->line( "filts_smem_loads", strprintf("filts_smem[%s] = filts[filts_off+(%%(tpb)*%s)];%s",ixe.c_str(),str(i).c_str(),eif.c_str()) );
      }
      // number of out chans per block; note: == work_out_chan_tile_dim*work_out_chan_dim
      uint32_t const filts_x_stride = rcg->get_arg_dims_by_name("filts").dstride("x");
      uint32_t const out_chan_bias_smem_load_iter = u32_ceil_div( filts_x_stride, rcg->rtc_call_geom.tpb );
      rcg->set( "out_chan_bias_smem_load_iter", str(out_chan_bias_smem_load_iter) );

      rcg->line( "biases_smem_loads","int32_t ocix; int32_t const ocix_base = %(GRP_ID_1D_out_chan_blk)*%(filts_x_stride);" );
      for( uint32_t i = 0; i != out_chan_bias_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	rcg->line( "biases_smem_loads", strprintf( "ocix = ocix_base + (%s %%%% %%(work_out_chan_tile_dim))*%%(work_out_chan_dim) + ( %s / %%(work_out_chan_tile_dim) );", ixe.c_str(), ixe.c_str() ) );
	if( (i+1)*rcg->rtc_call_geom.tpb > filts_x_stride ) {
	  rcg->line( "biases_smem_loads", "if( "+ixe+" < %(filts_x_stride) ) {" );eif = "}";}
	// note: load is (always) contiguous
	rcg->line( "biases_smem_loads", strprintf("if( ocix < %%(biases_out_chan_dim) ) {filts_smem[%s] = biases[ocix];}%s",ixe.c_str(),eif.c_str()) );
      }

    }

    void gen_op_conv( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      dims_t const & filts = rcg->get_arg_dims_by_name( "filts" );
      uint32_t const filts_smem_sz = filts.dstride("x");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      gen_filts_smem_loads( rcg, filts_smem_sz );

      uint32_t const pel_smem_load_iter = u32_ceil_div( (work.dsz( "pels" ) * work.dsz( "pels_tile" )), rcg->rtc_call_geom.tpb );
      rcg->set( "pel_smem_load_iter", str(pel_smem_load_iter) );
      rcg->set( "out_chan_tile",
		"(%(LOC_ID_1D_out_chan_tile)+%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim))");
      rcg->set( "pel_tile",
		"(%(LOC_ID_1D_pels_tile)+%(GRP_ID_1D_pels_blk)*%(work_pels_tile_dim))");
      rcg->set( "out_chan_ix","(%(out_chan_tile)*%(work_out_chan_dim))" );
      for( uint32_t i = 0; i != work.dsz( "pels" ); ++i ) {
	insert_nda_ix_exprs( rcg->tsvs, "pel_ix_" + str(i), must_find(rcg->all_ix_dims,"out_pel_ix"),
			     strprintf( "(%%(pel_tile)*%%(work_pels_dim)+%s)", str(i).c_str() ) );
      }
      for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem[%%(LOC_ID_1D_out_chan_tile)+%s*%%(work_out_chan_tile_dim)];",
				       str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem[%%(LOC_ID_1D_pels_tile)*%%(work_pels_dim)+%s];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }
      rcg->line( "stores", "int32_t tpix[%(work_pels_dim)];");
      rcg->line( "stores", "int32_t tcix[%(work_out_chan_dim)];");
      // FIXME: should somehow assert that both out_ix and pel_ix_N have the same dims here
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "stores",
		   strprintf( "tpix[%s] = %%(pel_ix_%s_img)*%%(out_img_stride) + "
			      "( %%(pel_ix_%s_x_nomod) %%%% (%%(out_y_dim)*%%(out_x_dim)) ); // cache out pel ixs ", // note: y:x adj-dim opt.
				str(ty).c_str(), str(ty).c_str(), str(ty).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "out_chan" ); ++ty ) {
	rcg->line( "stores", strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_chan_stride); // cache out chan ixs",
					  str(ty).c_str(), str(ty).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "stores", "if( %(pel_ix_"+str(ty)+"_x_nomod) >= %(pel_ix_0_dims_prod) ) { return; } "
		     "// this pel and the following are off-the-end pels, so don't store them." );
	for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];",
					  str((ty*work.dsz( "out_chan" )+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "stores", strprintf( "if( tcix[%s] < (%%(out_chan_dim)*%%(out_chan_stride)) ) { out[ tpix[%s] + tcix[%s] ] = %s; }",
					    str(tx).c_str(), str(ty).c_str(), str(tx).c_str(),
					  add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	}
      }
    }

    void gen_op_ipconv( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      //dims_t const & filts = get_arg_dims_by_name( "filts" );
      uint32_t const filts_smem_sz = work.dsz("out_chan_tile")*work.dsz("out_chan")*work.dsz("fioc_tile");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const out_chan_smem_load_iter = u32_ceil_div( filts_smem_sz, rcg->rtc_call_geom.tpb );
      for( uint32_t i = 0; i != out_chan_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string const filt_ix = "( LOC_ID_1D/%(work_fioc_tile_dim) + %(tpb)/%(work_fioc_tile_dim)* "+str(i)+")";
	string eif;
	// FIXME: can load garbage when ((out_chan_dim % filts_per_blk) != 0). pad output? add conditionals here? ignore?
	if( (i+1)*rcg->rtc_call_geom.tpb > filts_smem_sz ) {
	  rcg->line( "filts_smem_loads", "if( "+ixe+" < %(filts_smem_sz) ) {" );eif = "}";}
	rcg->line( "filts_smem_loads", strprintf("filts_smem[%s] = filts[filts_off+(%s*%%(filts_out_chan_stride))];%s",ixe.c_str(),filt_ix.c_str(),eif.c_str()) );
      }

      uint32_t const in_smem_sz = work.dsz("pels_tile")*work.dsz("pels")*work.dsz("fioc_tile");
      rcg->set( "in_smem_sz", str(in_smem_sz) );
      uint32_t const in_smem_load_iter = u32_ceil_div( in_smem_sz, rcg->rtc_call_geom.tpb );
      // currently, ipconv can only handle one output point per image, and assume the filt and in data-layouts are the
      // same (hence the name ipconv, for inner-product-conv).
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string const img_ix = "( LOC_ID_1D/%(work_fioc_tile_dim) + %(tpb)/%(work_fioc_tile_dim)* "+str(i)+")";
	string eif;
	// FIXME: can load garbage when ((in_img_dim % imgs_per_blk) != 0). pad input? add conditionals here? ignore?
	if( (i+1)*rcg->rtc_call_geom.tpb > in_smem_sz ) {
	  rcg->line( "in_smem_loads", "if( "+ixe+" < %(in_smem_sz) ) {" );eif = "}";}
	rcg->line( "in_smem_loads", strprintf("in_smem[%s] = in[in_off+(%s*%%(in_img_stride))];%s",ixe.c_str(),img_ix.c_str(),eif.c_str()) );
      }

      for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	rcg->line( "loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(work_fioc_tile_dim)];",
					 str(tx).c_str(), str(tx).c_str() ) );
      }
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) { // note: could merge with above loop, but we want to use ty for consistency
	rcg->line( "loads", strprintf( "in_strip[%s] = in_smem_off[%s*%%(work_fioc_tile_dim)];",
					 str(ty).c_str(), str(ty).c_str() ) );
      }
      rcg->line( "outs_to_filts_strip", "if( (in_pel+work_pel) >= %(in_img_dim) ) { return; } "
		   "// this pel and the following are off-the-end pels, so don't store them." );
      rcg->line( "outs_to_filts_strip", "switch(work_pel) { " );
      for( uint32_t ty = 0; ty != work.dsz( "pels" ); ++ty ) {
	rcg->line( "outs_to_filts_strip", "case "+str(ty)+":" );
	for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	  rcg->line( "fmas", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];",
					  str((ty*work.dsz( "out_chan" )+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  rcg->line( "outs_to_filts_strip", strprintf( "filts_strip[%s] = out_tile[%s];",
					    str(tx).c_str(), str((ty*work.dsz("out_chan")+tx)).c_str() ) );
	}
	rcg->line( "outs_to_filts_strip", "break;" );
      }
      rcg->line( "outs_to_filts_strip", "} " );

      for( uint32_t tx = 0; tx != work.dsz( "out_chan" ); ++tx ) {
	string ve = strprintf( "(filts_strip[%s] + biases[ocix+%s])", str(tx).c_str(), str(tx).c_str() );
	ve = rcg->op.get_u32("conv_has_relu") ? ( "max(0.0f,"+ve+")" ) : ve;
	for( uint32_t wb = work.dsz("fioc_tile") / 2; wb; wb /= 2 ) {
	  rcg->line( "stores", strprintf( "filts_strip[%s] += __shfl_down( filts_strip[%s], %s, %s );",
					    str(tx).c_str(), str(tx).c_str(), str(wb).c_str(),
					    str( work.dsz("fioc_tile") ).c_str() ) );
	}
	rcg->line( "stores", strprintf( "if( (%%(LOC_ID_1D_fioc_tile) == 0 ) && ((ocix + %s) < %%(out_chan_dim)) ) "
					  "{ out[out_off + %s*%%(out_chan_stride)] = %s; }",
					  str(tx).c_str(), str(tx).c_str(), str(ve).c_str() ) );
      }
    }

    string gva( uint32_t const & vw, uint32_t const & ix ) {
      assert_st( vw <= 16 );
      string const pss("0123456789abcdef");
      string const pss_alt("xyzw");
      string const maybe_s( (vw<=4) ? "" : "s" );
      return strprintf( "[%s].%s%s", str(ix/vw).c_str(), maybe_s.c_str(), string((vw<=4)?pss_alt:pss,ix%vw,1).c_str() );
    }

    void gen_op_sgemm_simd_local( rtc_call_gen_t * rcg ) {
      uint32_t const vw = rcg->op.get_u32( "vw" );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      assert( (work.dsz("Mt") % vw) == 0 );
      assert( (work.dsz("Nt") % vw) == 0 );
      dims_t const & a_dims = rcg->get_arg_dims_by_name("a");
      dims_t const & b_dims = rcg->get_arg_dims_by_name("b");
      uint64_t const blk_M = work.dsz("Mb")*work.dsz("Mt")/vw;
      uint32_t const a_sm_sz = blk_M*work.dsz("Kb");
      gen_sgemm_sm_load( rcg, "sm_loads", "a", a_sm_sz, blk_M, a_dims, vw );
      uint64_t const blk_N = work.dsz("Nb")*work.dsz("Nt")/vw;
      uint32_t const b_sm_sz = blk_N*work.dsz("Kb");
      gen_sgemm_sm_load( rcg, "sm_loads", "b", b_sm_sz, blk_N, b_dims, vw );


      for( uint32_t Kb = 0; Kb != work.dsz("Kb"); ++Kb ) {
	for( uint32_t Mt = 0; Mt != work.dsz("Mt")/vw; ++Mt ) {
	  rcg->line( "inner_loop_body", strprintf( "a_r[%s] = a_sm_off[%s];",
                                                   str(Mt).c_str(), str(Mt+Kb*blk_M).c_str() ) );
	}
	for( uint32_t Nt = 0; Nt != work.dsz("Nt")/vw; ++Nt ) {
	  rcg->line( "inner_loop_body", strprintf( "b_r[%s] = b_sm_off[%s];",
                                                   str(Nt).c_str(), str(Nt+Kb*blk_N).c_str() ) );
	}
        for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
          for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
            uint32_t const rix = (Mt*work.dsz("Nt")+Nt);
            rcg->line( "inner_loop_body", strprintf( "c_r[%s] += a_r%s*b_r%s;",str(rix).c_str(),
                                                     gva(vw,Mt).c_str(), gva(vw,Nt).c_str()));
          }
        }
      }
      rcg->line( "outs_to_b_r", "switch(Mt) { " );
      for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
	rcg->line( "outs_to_b_r", "case "+str(Mt)+":" );
        for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
          uint32_t const rix = (Mt*work.dsz("Nt")+Nt);
          rcg->line( "outs_to_b_r", strprintf( "b_r%s = c_r[%s];", gva(vw,Nt).c_str(), str(rix).c_str() ) );
	}
        rcg->line( "outs_to_b_r", "break;" );
      }
      rcg->line( "outs_to_b_r", "} " );

      // note: for this section, there will be a local 'Mt' in scope, used to adjust c_off each iteration
      for( uint32_t Nt = 0; Nt != work.dsz("Nt")/vw; ++Nt ) {
        rcg->line( "stores", strprintf( "((GASQ float%s *)c)[c_off+%s] = b_r[%s];",
                                        str(vw).c_str(), str(Nt).c_str(), str(Nt).c_str() ) );
      }
    }

    void gen_op_sgemm_simd( rtc_call_gen_t * rcg ) {
      uint32_t const vw = rcg->op.get_u32( "vw" );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      assert( (work.dsz("Mt") % vw) == 0 );
      assert( (work.dsz("Nt") % vw) == 0 );
      uint64_t a_K_stride = rcg->get_arg_dims_by_name("a").dstride("K");
      uint64_t b_K_stride = rcg->get_arg_dims_by_name("b").dstride("K");
      assert_st( ( a_K_stride % vw ) == 0 ); a_K_stride /= vw;
      assert_st( ( b_K_stride % vw ) == 0 ); b_K_stride /= vw;
      for( uint32_t Kb = 0; Kb != work.dsz("Kb"); ++Kb ) {
	for( uint32_t Mt = 0; Mt != work.dsz("Mt")/vw; ++Mt ) {
	  rcg->line( "inner_loop_body", strprintf( "a_r[%s] = ((GASQ float%s const *)a)[a_off+%s];",
                                                   str(Mt).c_str(), str(vw).c_str(), str(Mt+Kb*a_K_stride).c_str() ) );
	}
	for( uint32_t Nt = 0; Nt != work.dsz("Nt")/vw; ++Nt ) {
	  rcg->line( "inner_loop_body", strprintf( "b_r[%s] = ((GASQ float%s const *)b)[b_off+%s];",
                                                   str(Nt).c_str(), str(vw).c_str(), str(Nt+Kb*b_K_stride).c_str() ) );
	}
        for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
          for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
            uint32_t const rix = (Mt*work.dsz("Nt")+Nt);
            rcg->line( "inner_loop_body", strprintf( "c_r[%s] += a_r%s*b_r%s;",str(rix).c_str(),
                                                     gva(vw,Mt).c_str(), gva(vw,Nt).c_str()));
          }
        }
      }
      rcg->line( "outs_to_b_r", "switch(Mt) { " );
      for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
	rcg->line( "outs_to_b_r", "case "+str(Mt)+":" );
        for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
          uint32_t const rix = (Mt*work.dsz("Nt")+Nt);
          rcg->line( "outs_to_b_r", strprintf( "b_r%s = c_r[%s];", gva(vw,Nt).c_str(), str(rix).c_str() ) );
	}
        rcg->line( "outs_to_b_r", "break;" );
      }
      rcg->line( "outs_to_b_r", "} " );

      // note: for this section, there will be a local 'Mt' in scope, used to adjust c_off each iteration
      for( uint32_t Nt = 0; Nt != work.dsz("Nt")/vw; ++Nt ) {
        rcg->line( "stores", strprintf( "((GASQ float%s *)c)[c_off+%s] = b_r[%s];",
                                        str(vw).c_str(), str(Nt).c_str(), str(Nt).c_str() ) );
      }
    }

    void gen_op_sgemm_no_local( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      uint64_t const a_K_stride = rcg->get_arg_dims_by_name("a").dstride("K");
      uint64_t const b_K_stride = rcg->get_arg_dims_by_name("b").dstride("K");
      for( uint32_t Kb = 0; Kb != work.dsz("Kb"); ++Kb ) {
        for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
          rcg->line( "inner_loop_body", strprintf( "a_r[%s] = a[a_off+%s];", str(Mt).c_str(), str(Mt+Kb*a_K_stride).c_str() ) );
        }
        for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
          rcg->line( "inner_loop_body", strprintf( "b_r[%s] = b[b_off+%s];", str(Nt).c_str(), str(Nt+Kb*b_K_stride).c_str() ) );
        }
        for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
          for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
            uint32_t const rix = (Mt*work.dsz("Nt")+Nt);
            rcg->line( "inner_loop_body", strprintf( "c_r[%s] += a_r[%s]*b_r[%s];",str(rix).c_str(),
                                                     str(Mt).c_str(), str(Nt).c_str()));
          }
        }
      }
      gen_sgemm_write_out( rcg, 0 );
    }

    void gen_sgemm_sm_load( rtc_call_gen_t * rcg, string const & code_sec, string const & vn, uint64_t const & sm_sz,
                            uint64_t const row_len, dims_t const & vdims, uint32_t const & vw )
    {
      uint64_t stride = vdims.dstride("K");
      assert_st( ( stride % vw ) == 0 ); stride /= vw;
      uint64_t const rows = sm_sz / row_len;
      assert_st( row_len * rows == sm_sz ); // must be loading exactly an integer number of rows (not sensible otherwise?)
      assert_st( rows ); // no null load allowed (could just return no-op i suppose, if that ever made sense)
      assert_st( stride >= row_len ); // could relax in rows == 1 case (i.e. if there is no stride)
      uint64_t const row_pad = stride - row_len;

      string const smvn = vn + "_sm"; // note: could be customizable.
      rcg->set( smvn + "_sz", str(sm_sz) );
      uint32_t const sm_load_iter = u32_ceil_div( sm_sz, rcg->rtc_call_geom.tpb );
      for( uint32_t i = 0; i != sm_load_iter; ++i ) {
        uint64_t const iter_sm_off = rcg->rtc_call_geom.tpb*i;
        uint64_t const iter_row = iter_sm_off / row_len;
        uint64_t const iter_row_off = iter_sm_off % row_len;
        uint64_t const iter_off = iter_sm_off + iter_row*row_pad;

	string extra_off_str;
        if( row_pad && (iter_row_off + rcg->rtc_call_geom.tpb > row_len) ) { // more than one row-per-block, need to add dynamic offset term
          extra_off_str = strprintf("+(LOC_ID_1D+%s)/%s*%s", str(iter_row_off).c_str(),
                                    str(row_len).c_str(), str(row_pad).c_str() );
        }

	string eif;
	if( (iter_sm_off + rcg->rtc_call_geom.tpb) > sm_sz ) {  // block extends past end of sm, need to add guard
	  rcg->line( code_sec, strprintf("if( (LOC_ID_1D+%s) < %s ) {", str(iter_sm_off).c_str(), str(sm_sz).c_str() ) ); eif="}";}
        string vn_vw = vn;
        assert_st( vw );
        if( vw > 1 ) { vn_vw = strprintf("((GASQ %%(a_tn)%s const *)(%s))", str(vw).c_str(), vn.c_str() ); }
        assert_st( vdims.tinfo().is_float == 1 );
        bool const half_to_float = (vdims.tsz() == 2);
        if( half_to_float ) {
          rcg->line( code_sec, strprintf("%s[LOC_ID_1D+%s] = vload_half( %s_off+%s%s, %s );%s",
                                         smvn.c_str(), str(iter_sm_off).c_str(),
                                         vn.c_str(), str(iter_off).c_str(),
                                         extra_off_str.c_str(),
                                         vn_vw.c_str(),
                                         eif.c_str()) );
        } else {
          assert_st( vdims.tsz() == 4 ); // i.e. float
          rcg->line( code_sec, strprintf("%s[LOC_ID_1D+%s] = %s[%s_off+%s%s];%s",
                                         smvn.c_str(), str(iter_sm_off).c_str(),
                                         vn_vw.c_str(), vn.c_str(), str(iter_off).c_str(),
                                         extra_off_str.c_str(), eif.c_str()) );
        }
      }
    }

    void gen_op_sgemm( rtc_call_gen_t * rcg ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      uint64_t const blk_M = work.dsz("Mb")*work.dsz("Mt");
      uint32_t const a_sm_sz = blk_M*work.dsz("Kb");
      gen_sgemm_sm_load( rcg, "sm_loads", "a", a_sm_sz, blk_M, rcg->get_arg_dims_by_name("a"), 1 );
      uint64_t const blk_N = work.dsz("Nb")*work.dsz("Nt");
      uint32_t const b_sm_sz = blk_N*work.dsz("Kb");
      gen_sgemm_sm_load( rcg, "sm_loads", "b", b_sm_sz, blk_N, rcg->get_arg_dims_by_name("b"), 1 );

      for( uint32_t Kb = 0; Kb != work.dsz("Kb"); ++Kb ) {
	for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
	  rcg->line( "inner_loop_body", strprintf( "a_r[%s] = a_sm_off[%s];", str(Mt).c_str(), str(Mt+Kb*blk_M).c_str()));
	}
	for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
	  rcg->line( "inner_loop_body", strprintf( "b_r[%s] = b_sm_off[%s];", str(Nt).c_str(), str(Nt+Kb*blk_N).c_str()));
	  //rcg->line( "inner_loop_body", strprintf( "b_r[%s] = b[k*%%(b_K_stride)+thr_N+%s];", str(Nt).c_str(), str(Nt).c_str() ) );
	}
	for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
          for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
            uint32_t const rix = (Mt*work.dsz("Nt")+Nt);
            rcg->line( "inner_loop_body", strprintf( "c_r[%s] += a_r[%s]*b_r[%s];",str(rix).c_str(),
                                                     str(Mt).c_str(), str(Nt).c_str()));
          }
        }
      }
      dims_t const & c_dims = rcg->get_arg_dims_by_name("c");
      assert( c_dims.tinfo().is_float );
      bool const half_to_float = (c_dims.tsz() == 2);
      assert( half_to_float || (c_dims.tsz() == 4) );
      gen_sgemm_write_out( rcg, half_to_float );
    }
    void gen_sgemm_write_out( rtc_call_gen_t * rcg, bool const & half_to_float ) {
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      rcg->line( "outs_to_b_r", "switch(Mt) { " );
      for( uint32_t Mt = 0; Mt != work.dsz("Mt"); ++Mt ) {
	rcg->line( "outs_to_b_r", "case "+str(Mt)+":" );
        for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
          uint32_t const rix = (Mt*work.dsz("Nt")+Nt);
          rcg->line( "outs_to_b_r", strprintf( "b_r[%s] = c_r[%s];", str(Nt).c_str(), str(rix).c_str() ) );
	}
        rcg->line( "outs_to_b_r", "break;" );
      }
      rcg->line( "outs_to_b_r", "} " );

      // note: for this section, there will be a local 'Mt' in scope, used to adjust c_off each iteration
      for( uint32_t Nt = 0; Nt != work.dsz("Nt"); ++Nt ) {
        if( half_to_float ) {
          rcg->line( "stores", strprintf( "vstore_half( b_r[%s], c_off+%s, c );", str(Nt).c_str(), str(Nt).c_str() ) );
        } else {
          rcg->line( "stores", strprintf( "c[c_off+%s] = b_r[%s];", str(Nt).c_str(), str(Nt).c_str() ) );
        }
      }
    }

    void gen_op_k1conv_simd( rtc_call_gen_t * rcg ) {
      uint32_t const vw = rcg->op.get_u32( "vw" );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      assert( (work.dsz("pels") % vw) == 0 );
      assert( (work.dsz("out_chan") % vw) == 0 );
      uint64_t in_chan_stride = rcg->get_arg_dims_by_name("in").dstride("chan");
      uint64_t filts_in_chan_stride = rcg->get_arg_dims_by_name("filts").dstride("in_chan");
      assert_st( ( in_chan_stride % vw ) == 0 ); in_chan_stride /= vw;
      assert_st( ( filts_in_chan_stride % vw ) == 0 ); filts_in_chan_stride /= vw;

      uint32_t const Kb_dim = rcg->op.get_u32( "Kb" );
      for( uint32_t Kb = 0; Kb != Kb_dim; ++Kb ) {
	for( uint32_t tx = 0; tx != work.dsz("pels")/vw; ++tx ) {
          rcg->line( "inner_loop_body", strprintf( "in_strip[%s] = ((GASQ float%s const *)in)[in_off+%s];",
                                                   str(tx).c_str(), str(vw).c_str(), str(tx+Kb*in_chan_stride).c_str() ) );
        }
	for( uint32_t ty = 0; ty != work.dsz("out_chan")/vw; ++ty ) {
          rcg->line( "inner_loop_body", strprintf( "filts_strip[%s] = ((GASQ float%s const *)filts)[filts_off+%s];",
                                                   str(ty).c_str(), str(vw).c_str(), str(ty+Kb*filts_in_chan_stride).c_str() ) );
        }
	for( uint32_t tx = 0; tx != work.dsz("pels"); ++tx ) {
          for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) {
            uint32_t const rix = (tx*work.dsz("out_chan")+ty);
            rcg->line( "inner_loop_body", strprintf( "out_tile[%s] += in_strip%s*filts_strip%s;",str(rix).c_str(),
                                                     gva(vw,tx).c_str(), gva(vw,ty).c_str()));
          }
        }
      }

      rcg->line( "outs_to_in_strip", "switch(ty) { " );
      for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) {
	rcg->line( "outs_to_in_strip", "case "+str(ty)+":" );
        for( uint32_t tx = 0; tx != work.dsz("pels"); ++tx ) {
          uint32_t const rix = (tx*work.dsz("out_chan")+ty);
          string const ve = strprintf( "(out_tile[%s]+filts_strip%s)", str(rix).c_str(), gva(vw,ty).c_str() );
          rcg->line( "outs_to_in_strip", strprintf( "in_strip%s = %s;",
                                                    gva(vw,tx).c_str(), maybe_add_relu(rcg,ve).c_str() ) );
	}
        rcg->line( "outs_to_in_strip", "break;" );
      }
      rcg->line( "outs_to_in_strip", "} " );

      // note: for this section, there will be a local 'tx' in scope
      for( uint32_t tx = 0; tx != work.dsz("pels")/vw; ++tx ) {
        rcg->line( "stores", strprintf( "((GASQ float%s *)out)[out_off+%s] = in_strip[%s];",
                                        str(vw).c_str(), str(tx).c_str(), str(tx).c_str() ) );
      }

    }

    void gen_op_conv_simd( rtc_call_gen_t * rcg ) {
      uint32_t const vw = rcg->op.get_u32( "vw" );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      dims_t const & in_pels = rcg->get_arg_dims_by_name( "in_pels" );
      assert( (work.dsz("pels") % vw) == 0 );
      assert( (work.dsz("out_chan") % vw) == 0 );
      //uint64_t in_chan_stride = rcg->get_arg_dims_by_name("in").dstride("chan");
      //printf( "in_chan_stride=%s\n", str(in_chan_stride).c_str() );
      uint64_t filts_in_chan_stride = rcg->get_arg_dims_by_name("filts").dstride("in_chan");
      //assert_st( ( in_chan_stride % vw ) == 0 ); in_chan_stride /= vw;
      assert_st( ( filts_in_chan_stride % vw ) == 0 ); filts_in_chan_stride /= vw;

      dims_t const & stride = rcg->get_arg_dims_by_name( "stride" );

      uint32_t const row_extra_off = (stride.dsz("y")-1)*in_pels.dstride("y");
      assert_st( stride.dsz("y") == stride.dsz("x") ); // FIXME/NOTE: see uniform stride FIXMEs in kernel and cnn_op.cc

      uint32_t const Kb_dim = rcg->op.get_u32( "Kb" );
      for( uint32_t Kb = 0; Kb != Kb_dim; ++Kb ) {
	for( uint32_t tx = 0; tx != work.dsz("pels"); ++tx ) {
          assert_st( Kb == 0 ); // see FIXME in cnn_op.cc; stride in in is wrong here (needs to be per X/Y/chan,
                                // prob. need other approach like fixed unroll over x dim or the like to be eff.)
          string reo;
          if( row_extra_off ) { reo = strprintf( "+((out_x + %s)/%%(out_pels_x_dim)*%s)", str(tx).c_str(),str(row_extra_off).c_str() ); }
          rcg->line( "inner_loop_body", strprintf( "in_strip%s = in[in_off+%s %s];",  // FIXME: use Kb
                                                   gva(vw,tx).c_str(), str(tx*stride.dsz("x")).c_str(), reo.c_str() ) );
        }
	for( uint32_t ty = 0; ty != work.dsz("out_chan")/vw; ++ty ) {
          rcg->line( "inner_loop_body", strprintf( "filts_strip[%s] = ((GASQ float%s const *)filts)[filts_off+%s];",
                                                   str(ty).c_str(), str(vw).c_str(), str(ty+Kb*filts_in_chan_stride).c_str() ) );
        }
	for( uint32_t tx = 0; tx != work.dsz("pels"); ++tx ) {
          for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) {
            uint32_t const rix = (tx*work.dsz("out_chan")+ty);
            rcg->line( "inner_loop_body", strprintf( "out_tile[%s] += in_strip%s*filts_strip%s;",str(rix).c_str(),
                                                     gva(vw,tx).c_str(), gva(vw,ty).c_str()));
          }
        }
      }

      rcg->line( "outs_to_in_strip", "switch(ty) { " );
      for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) {
	rcg->line( "outs_to_in_strip", "case "+str(ty)+":" );
        for( uint32_t tx = 0; tx != work.dsz("pels"); ++tx ) {
          uint32_t const rix = (tx*work.dsz("out_chan")+ty);
          string const ve = strprintf( "(out_tile[%s]+filts_strip%s)", str(rix).c_str(), gva(vw,ty).c_str() );
          rcg->line( "outs_to_in_strip", strprintf( "in_strip%s = %s;",
                                                    gva(vw,tx).c_str(), maybe_add_relu(rcg,ve).c_str() ) );
	}
        rcg->line( "outs_to_in_strip", "break;" );
      }
      rcg->line( "outs_to_in_strip", "} " );

      // note: for this section, there will be a local 'tx' in scope
      for( uint32_t tx = 0; tx != work.dsz("pels")/vw; ++tx ) {
        rcg->line( "stores", strprintf( "((GASQ float%s *)out)[out_off+%s] = in_strip[%s];",
                                        str(vw).c_str(), str(tx).c_str(), str(tx).c_str() ) );
      }

    }

    void gen_op_k1conv( rtc_call_gen_t * rcg ) {
      assert_st( get_xy_dims( rcg->get_arg_dims_by_name( "in_pad" ) ).is_zeros() );
      assert_st( (get_xy_dims( rcg->get_arg_dims_by_name( "stride" ) ) == u32_pt_t{1,1}) );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      dims_t const & filts = rcg->get_arg_dims_by_name( "filts" );
      assert_st( filts.dsz("x") == 1 ); assert_st( filts.dsz("y") == 1 );
      dims_t const & in = rcg->get_arg_dims_by_name( "in" );
      dims_t const & out = rcg->get_arg_dims_by_name( "out" );
      // calculate needed smem sizes (and total kernel needed smem size)
      // note: filts and in smem are used concurrently, then just all of all_smem as an output buffer
      uint32_t const filts_smem_sz = filts.dstride("in_chan")*in.dsz("blk_iter_chan");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      uint32_t const out_smem_sz = work.dsz("pels_tile")*work.dsz("out_chan_tile")*work.dsz("pels"); // note: == oi->tpb*work.dsz("pels")
      rcg->set( "out_smem_sz", str(out_smem_sz) ); // note: unused, but assumed that all_smem_sz >= out_smem_sz
      uint32_t const all_smem_sz = std::max( out_smem_sz, filts_smem_sz+in.dstride("blk_iter") ); // note: %(in_blk_iter_sz) == in_smem_sz
      rcg->set( "all_smem_sz", str(all_smem_sz) );

      // generate smem loads
      gen_filts_smem_loads( rcg, filts_smem_sz );
      uint32_t const in_smem_load_iter = u32_ceil_div( in.dstride("blk_iter"), rcg->rtc_call_geom.tpb );
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rcg->rtc_call_geom.tpb > in.dstride("blk_iter") ) { rcg->line( "smem_loads", "if( "+ixe+" < %(in_blk_iter_stride)) { ");eif = "}";}
	rcg->line( "smem_loads", strprintf("    in_smem[%s] = in[ blk_in_ix_base + (%%(tpb)*%s) ];%s\n",
					     ixe.c_str(),str(i).c_str(),eif.c_str()) );
      }
      rcg->set( "out_chan_tile", "(%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim)+%(LOC_ID_1D_out_chan_tile))");
      rcg->set( "out_chan_ix","(%(out_chan_tile)*%(work_out_chan_dim))" );

      // rcg->line( "stores", "  if( %(out_line_img) >= %(out_ix_img_dim) ) { return; } "; // not possible due to no-partial-imgs-per-block
      // FIXME: should somehow assert that both out_ix and pel_ix_N have the same dims here
      // FIXME: out_pel must be per-tpix (again)
      if( out.get_dim_by_name("blk") ) { // aka if(write_xposed) -- if this dim names exists in the output, we know to write in xposed format
	// padded # of in chans of next layer  == out.dsz("blk_iter")*out.dsz("blk_iter_chan")
	// padded # of out chans of this layer == work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan")
	// if these are ==, we don't have to worry about bounds-checking our writes to out in the chan dim
	assert_st( work.dsz("out_chan_blk")*work.dsz("out_chan_tile")*work.dsz("out_chan") == out.dsz("blk_iter")*out.dsz("blk_iter_chan") );
	// padded # of in pels of next layer:  == out.dsz("blk")*out.dsz("blk_pel")
	// padded # of out pels of this layer: == work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels")
	// if these are ==, we don't have to worry about bounds-checking our writes to out in the pel dim
	assert_st( work.dsz("pels_blk")*work.dsz("pels_tile")*work.dsz("pels") == out.dsz("blk")*out.dsz("blk_pel") );

	// we assume out_blk_pel_dim (== noi->thr_per_blk.d[0]*t_tile_sz) is divisible by t_tile_sz. but let's check it explicitly:
	// FIXME_WXP: i don't see where we assume this, and hence i dunno what t_tile_sz refers to below. poop. assert is removed for now:
	// assert_st( (out.dsz("blk_pel") % t_tile_sz) == 0 );
	// we assume the out chans are a single (span of) dims in out. FIXME: check this?. FIXME_WXP: what does this even mean?

	//rcg->line( "stores", "  int32_t xpbuf[%(work_out_chan_dim)];\n";
	// FIXME: assumes (for GRP_ID_1D_pels_blk*... term) that input and output block have same # of pels ... too strong?
	assert_st( out.dsz("blk_pel") == in.dsz("blk_pel") );
	rcg->line( "stores", "int32_t const out_ix = (%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim)*%(work_out_chan_dim))*%(out_blk_iter_chan_stride) + %(GRP_ID_1D_pels_blk)*%(out_blk_stride);" );
	rcg->line( "stores", "int32_t xpbuf_rd_pel;" );
	rcg->line( "stores", "int32_t xpbuf_rd_chan;" );

	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  // transpose each thread's tx'th out_chan (= work_out_chan_dim out chans across all threads) into xpbuf (again across all threads)
	  // such that we can do (mostly) sequential writes to global memory for this set of work_out_chan_dim out chans
	  rcg->line( "stores", "  BARRIER_SYNC;" );
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // out_tile[] (registers) -> all_smem[]
	    rcg->line( "stores", strprintf( "out_smem_off[%%(tpb)*%s] = %s;", str(ty).c_str(),
					    add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	  }
	  rcg->line( "stores", "  BARRIER_SYNC;" );
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // all_smem[] -> [xpbuf[] (registers)] -> out[] (global)
	    // here, we reshape the threads so that the total threads across iterations (%(tbp)*work.dsz("pels")) covers
	    // the space of the data in out_smem as a (simple) 2D array ordered as chan:pel. thus, for each thread, we
	    // have a single chan and pel index that we must read from smem and write to global memory. this mapping is
	    // such that the writes to global memory are somewhat sequential (with jumps at chan boundaries). however,
	    // for the reads from smem we just calculate the correct index and hope for the best. note that the actual
	    // output chan indexes read/written to here are strided by %(work_out_chan_dim) and offset by tx.
	    string const obe = "(LOC_ID_1D + %(tpb)*"+str(ty)+")";
	    rcg->line( "stores", "  xpbuf_rd_pel = "+obe+" %% %(out_blk_pel_dim) ;" );
	    rcg->line( "stores", "  xpbuf_rd_chan = "+obe+" / %(out_blk_pel_dim) ;" );
	    rcg->line( "stores", strprintf( "out[out_ix + xpbuf_rd_pel + (xpbuf_rd_chan*%%(work_out_chan_dim)+%s)*%%(out_blk_iter_chan_stride)] = "
					      "all_smem[xpbuf_rd_chan+(xpbuf_rd_pel %%%% %%(work_pels_dim))*%%(tpb)"
					      "+ (xpbuf_rd_pel / %%(work_pels_dim))*%%(work_out_chan_tile_dim) ];",
					      str(tx).c_str() ) );
	  }
	  for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) { // xpbuf[] registers -> out[] (global)
	    // TODO/UNUSED?
	  }
	}
      } else {
	rcg->line( "stores", "  int32_t tpix[%(work_pels_dim)];" );
	rcg->line( "stores", "  int32_t tcix[%(work_out_chan_dim)];" );
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  insert_nda_ix_exprs( rcg->tsvs, "out_pel_" + str(ty), must_find(rcg->all_ix_dims,"out_ref_pel"),
			       "( (%(GRP_ID_1D_pels_blk)*%(work_pels_tile_dim) + %(LOC_ID_1D_pels_tile))*%(work_pels_dim) + "+str(ty)+" )" );
	  rcg->line( "stores", strprintf( "  tpix[%s] = %%(out_pel_%s_img)*%%(out_img_stride) + "
					    " %%(out_pel_%s_x)*%%(out_x_stride) + %%(out_pel_%s_y)*%%(out_y_stride) " // FIXME_WXP:restore: y:x adj-dim opt?
					    "  ; // cache out pel ixs",
					    str(ty).c_str(), str(ty).c_str(), str(ty).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("out_chan"); ++ty ) {
	  rcg->line( "stores", strprintf( "  tcix[%s] = (%%(out_chan_ix)+%s)*%%(out_chan_stride); // cache out chan ixs",
					    str(ty).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  rcg->line( "stores", "  if( %(out_pel_"+str(ty)+"_img) >= %(out_img_dim) ) { return; } "
		       "// this pel and the following are off-the-end pels, so don't store them." );
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    rcg->line( "stores", strprintf( "if( tcix[%s] < (%%(out_chan_dim)*%%(out_chan_stride)) ) { out[ tpix[%s] + tcix[%s] ] = %s; }",
					      str(tx).c_str(), str(ty).c_str(), str(tx).c_str(),
					    add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	  }
	}
      }
      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  rcg->line( "dummy_stores", strprintf( "out_off[%s] = %s;",
						  str((ty*work.dsz("out_chan")+tx)*rcg->rtc_call_geom.tpb).c_str(),
						add_bias_then_maybe_relu(rcg,work,tx,ty).c_str() ) );
	}
      }
      for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	rcg->line( "bias_loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(work_out_chan_tile_dim)];",
					      str(tx).c_str(), str(tx).c_str() ) );
      }
      assert_st( in.dsz("blk_pel") == work.dsz("pels_tile")*work.dsz("pels") ); // by input xform design
      for( uint32_t ict = 0; ict != in.dsz("blk_iter_chan"); ++ict ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  rcg->line( "inner_loop_body", strprintf( "filts_strip[%s] = filts_smem_off[(%s*%%(filts_in_chan_stride))+%s*%%(work_out_chan_tile_dim)];",
						     str(tx).c_str(), str(ict).c_str(), str(tx).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  rcg->line( "inner_loop_body", strprintf( "in_strip[%s] = in_smem_off[(%s*%%(in_blk_pel_dim)+%s)];",
						     str(ty).c_str(), str(ict).c_str(), str(ty).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    rcg->line( "inner_loop_body", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];",
						       str((ty*work.dsz("out_chan")+tx)).c_str(), str(tx).c_str(), str(ty).c_str() ) );
	  }
	}
      }
    }

    void gen_op_tconv( rtc_call_gen_t * rcg ) {
      dims_t const & stride = rcg->get_arg_dims_by_name( "stride" );
      dims_t const & work = rcg->get_arg_dims_by_name( "work" );
      dims_t const & filts = rcg->get_arg_dims_by_name( "filts" );
      dims_t const & in = rcg->get_arg_dims_by_name( "in" );
      uint32_t const filts_smem_sz = filts.dstride("y");
      rcg->set( "filts_smem_sz", str(filts_smem_sz) );
      gen_filts_smem_loads( rcg, filts_smem_sz );
      rcg->line( "filts_smem_loads", "filts_off += %(filts_smem_sz);" );
      uint32_t const in_smem_load_iter = u32_ceil_div( in.dstride("blk_in_chan"), rcg->rtc_call_geom.tpb );  // in smem loads
      for( uint32_t i = 0; i != in_smem_load_iter; ++i ) {
	string const ixe = "(LOC_ID_1D + %(tpb) * "+str(i)+")";
	string eif;
	if( (i+1)*rcg->rtc_call_geom.tpb > in.dstride("blk_in_chan") ) { rcg->line( "in_smem_loads", "if( "+ixe+" < %(in_blk_in_chan_stride)) { " );eif = "}";}
	rcg->line( "in_smem_loads", strprintf("in_smem[%s] = in[ blk_in_ix_base + (%%(tpb)*%s) ];%s",	ixe.c_str(),str(i).c_str(),eif.c_str()));
      }
      rcg->line( "in_smem_loads", "blk_in_ix_base += %(in_blk_in_chan_stride);" );

      for( uint32_t i = 0; i != in.dsz("blk_x"); ++i ) {
	rcg->line( "inner_loop_body", strprintf( "in_strip[%s] = in_smem_off[%s];", str(i).c_str(), str(i).c_str() ) );
      }
      assert_st( work.dsz("out_chan_tile") == filts.dsz("out_chan_tile") ); // also == %(filts_out_chan_reg_stride)
      for( uint32_t kx = 0; kx != filts.dsz("x"); ++kx ) {
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	  rcg->line( "inner_loop_body", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(filts_x_stride)+%s*%%(filts_out_chan_reg_stride)];",
						     str(tx).c_str(), str(kx).c_str(), str(tx).c_str() ) );
	}
	for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	  for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	    rcg->line( "inner_loop_body", strprintf( "out_tile[%s] += filts_strip[%s]*in_strip[%s];",
						     str((ty*work.dsz("out_chan")+tx)).c_str(),
						     str(tx).c_str(), str(ty*stride.dsz("x")+kx).c_str()));
	  }
	}
      }
      for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
	rcg->line( "bias_loads", strprintf( "filts_strip[%s] = filts_smem_off[%s*%%(filts_out_chan_reg_stride)];", str(tx).c_str(), str(tx).c_str() ) );
      }
      //rcg->line( "stores", "  if( %(out_line_y) >= %(out_ix_y_stride) ) { return; }" ); // not possible
      rcg->line( "stores", "if( %(out_line_img) >= %(out_img_dim) ) { return; }" );
      rcg->line( "stores", "int32_t out_x = %(GRP_ID_1D_blk_bx)*%(work_pels_dim);" );
      rcg->line( "stores", "int32_t out_chan = (%(GRP_ID_1D_out_chan_blk)*%(work_out_chan_tile_dim) + %(LOC_ID_1D_out_chan_tile))*%(work_out_chan_dim);" );
      rcg->line( "stores", "GASQ float * out_off = out + %(out_line_img)*%(out_img_stride) + out_chan*%(out_chan_stride) + "
		   "%(out_line_y)*%(out_y_stride) + out_x*%(out_x_stride) ;" );

      for( uint32_t ty = 0; ty != work.dsz("pels"); ++ty ) {
	rcg->line( "stores", "if( (out_x + "+str(ty)+") >= %(out_x_dim) ) { return; } "
		     "// this x value and the following are off-the-end pels, so don't store them." );
	for( uint32_t tx = 0; tx != work.dsz("out_chan"); ++tx ) {
#if 1
	  string const ve = add_bias_then_maybe_relu(rcg,work,tx,ty);

#else
	  string const ve = strprintf( "(filts_strip[%s])", str(tx).c_str() );
#endif
	  rcg->line( "stores", strprintf( "if( (out_chan + %s) < %%(out_chan_dim) ) { "
						   "out_off[ %s*%%(out_chan_stride) + %s*%%(out_x_stride) ] = %s; }",
						   str(tx).c_str(), str(tx).c_str(), str(ty).c_str(), ve.c_str() ) );
	}
      }
    }
  };

  p_custom_codegen_t make_cnn_custom_codegen_t( void ) { return p_custom_codegen_t( new cnn_custom_codegen_t ); }

}
