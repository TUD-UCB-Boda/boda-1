// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"geom_prim.H"
#include"disp_util.H"
#include<SDL.h>
#include<SDL_ttf.h>
#include"timers.H"
#include"str_util.H"
#include"img_io.H"
#include<poll.h>
#include"mutex.H"
#include"asio_util.H"
#include"anno_util.H"
#include"lexp.H"
#include"nesi.H"

namespace boda 
{
  void layout_elem_t::calc_layout( map_str_u32_pt_t const & leaf_szs, map_str_u32_pt_t & leaf_ncs ) {
    set_size( leaf_szs );
    set_leaf_ncs( u32_pt_t(), leaf_ncs );
  }

  string const leaf_prefix = "leaf:";

  void layout_elem_t::set_size( map_str_u32_pt_t const & leaf_szs ) {
    for( vect_p_layout_elem_t::const_iterator i = kids.begin(); i != kids.end(); ++i ) {
      (*i)->set_size( leaf_szs );
    }
    assert_st( sz == u32_pt_t() ); // should be unset/zero
    string maybe_leaf_id = mode;
    if( maybe_strip_prefix( maybe_leaf_id, leaf_prefix ) ) {
      if( !kids.empty() ) { rt_err("leaf layout node must not have kids"); }
      sz = must_find( leaf_szs, maybe_leaf_id );
    } else if( mode == "horiz" || mode == "vert" ) {
      bool d = (mode == "horiz");
      for( vect_p_layout_elem_t::const_iterator i = kids.begin(); i != kids.end(); ++i ) {
        max_eq( sz.d[d], (*i)->sz.d[d] );
      }
      for( vect_p_layout_elem_t::const_iterator i = kids.begin(); i != kids.end(); ++i ) {
        (*i)->rel_nc = u32_pt_t{ (sz.d[d] - (*i)->sz.d[d]) / 2, sz.d[!d] };
        if( d ) { (*i)->rel_nc.dims_swap(); }
        sz.d[!d] += (*i)->sz.d[!d];
      }
    } else if( mode == "grid" ) {
      rt_err( "grid layout mode TODO" );
    } else { rt_err( "unknown layout_mode=" + mode ); }
  }

  void layout_elem_t::set_leaf_ncs( u32_pt_t path_sum_nc, map_str_u32_pt_t & leaf_ncs ) {
    path_sum_nc += rel_nc;
    for( vect_p_layout_elem_t::const_iterator i = kids.begin(); i != kids.end(); ++i ) {
      (*i)->set_leaf_ncs( path_sum_nc, leaf_ncs );
    }
    string maybe_leaf_id = mode;
    if( maybe_strip_prefix( maybe_leaf_id, leaf_prefix ) ) {
      assert_st( kids.empty() ); // check in set_size()
      must_insert( leaf_ncs, maybe_leaf_id, path_sum_nc );
    }
  }
  
  SDL_Rect box_to_sdl( i32_box_t const & b ) {
    i32_pt_t const bsz = b.sz();
    i32_pt_t const & bnc = b.p[0];
    return SDL_Rect{bnc.d[0],bnc.d[1],bsz.d[0],bsz.d[1]};
  }
  i32_box_t box_from_sdl( SDL_Rect const & b ) { return i32_box_t{{b.x,b.y},{b.x+b.w,b.y+b.h}}; }

  SDL_Color color_to_sdl( uint32_t const & c ) { return SDL_Color{ get_chan(0,c),get_chan(1,c),get_chan(2,c),get_chan(3,c)}; }
  void sdl_set_color_from_pel( p_SDL_Renderer const & r, uint32_t const & c ) {
    SDL_SetRenderDrawColor( r.get(), get_chan(0,c),get_chan(1,c),get_chan(2,c),get_chan(3,c) ); }

  void rt_err_sdl( char const * const msg ); // like rt_err, but prints sdl error
  void rt_err_sdl( char const * const msg ) { rt_err( strprintf( "%s (SDL error: %s)", msg, SDL_GetError() ) ); }

#define DECL_MAKE_P_SDL_OBJ( tn ) p_SDL_##tn make_p_SDL( SDL_##tn * const rp ) { return p_SDL_##tn( rp, SDL_Destroy##tn ); }

  DECL_MAKE_P_SDL_OBJ( Window );
  DECL_MAKE_P_SDL_OBJ( Renderer );
  DECL_MAKE_P_SDL_OBJ( Texture );

#undef DECL_MAKE_P_SDL_OBJ

  struct YV12_buf_t {
    p_uint8_t d;
    uint32_t w;
    uint32_t h;

    // calculated fields
    uint32_t sz;
    uint8_t * Y;
    uint8_t * V;
    uint8_t * U;

    YV12_buf_t( void ) : w(0), h(0), sz(0), Y(0), V(0), U(0) { }
    void set_sz_and_alloc( u32_pt_t const & sz_ ) {
      w = sz_.d[0]; assert_st( !(w&1) );
      h = sz_.d[1]; assert_st( !(h&1) );
      sz = w * ( h + (h/2) );
      d = ma_p_uint8_t( sz, 4096 );
      for( uint32_t i = 0; i < sz; ++i ) { d.get()[i] = 128; } // init to grey
      Y = d.get();
      V = Y + w*h; // w*h == size of Y
      U = V + (w/2)*(h/2); // (w/2)*(h/2) == size of V (and U, which is unneeded)
    }

    void YVUat( uint8_t * & out_Y, uint8_t * & out_V, uint8_t * & out_U, uint32_t const & x, uint32_t const & y ) const { 
      assert_st( x < w ); assert_st( y < h );
      out_Y = Y + y*w + x; out_V = V + (y>>1)*(w>>1) + (x>>1); out_U = U + (y>>1)*(w>>1) + (x>>1); 
    }

    p_img_t as_img_t_yuv_borrowed( void ) const {
      vect_p_nda_uint8_t yuv_ndas;
      yuv_ndas.push_back( make_shared<nda_uint8_t>( dims_t{ vect_uint32_t{uint32_t(h), uint32_t(w)}, vect_string{ "y","x" },"uint8_t" },
                                                    p_uint8_t( Y, null_deleter<uint8_t>() ) ) );
      yuv_ndas.push_back( make_shared<nda_uint8_t>( dims_t{ vect_uint32_t{uint32_t(h/2), uint32_t(w/2)}, vect_string{ "y","x" },"uint8_t" },
                                                    p_uint8_t( U, null_deleter<uint8_t>() ) ) );
      yuv_ndas.push_back( make_shared<nda_uint8_t>( dims_t{ vect_uint32_t{uint32_t(h/2), uint32_t(w/2)}, vect_string{ "y","x" },"uint8_t" },
                                                    p_uint8_t( V, null_deleter<uint8_t>() ) ) );
      p_img_t ret = make_shared< img_t >();
      ret->set_sz_and_pels_from_yuv_420_planes( yuv_ndas );
      return ret;
    }
    
  };

  void img_to_YV12( YV12_buf_t const & YV12_buf, p_img_t const & img, uint32_t const out_x, uint32_t const out_y ) {
    uint32_t const w = img->sz.d[0]; 
    uint32_t const h = img->sz.d[1];
    
    uint8_t *out_Y, *out_V, *out_U;
    if( !img->yuv_pels.empty() ) {
      uint8_t const * yr;
      uint8_t const * ur;
      uint8_t const * vr;
      for( uint32_t y = 0; y < h; ++y ) {
        YV12_buf.YVUat( out_Y, out_V, out_U, out_x, out_y+y );
        img->get_YUV_row_addr( y, yr, ur, vr );
        std::copy( yr, yr+w, out_Y );
        if( !(y&1) ) { // for even rows, copy u/v data too
          std::copy( ur, ur+(w+1)/2, out_U );          
          std::copy( vr, vr+(w+1)/2, out_V );
        }
      }
    } else if( img->pels ) {
      for( uint32_t y = 0; y < h; ++y ) {
        YV12_buf.YVUat( out_Y, out_V, out_U, out_x, out_y+y );
        uint32_t const * rgb = img->get_row_addr( y );
        for( uint32_t x = 0; x < w; ++x, ++rgb ) {
          rgba2y( *rgb, *(out_Y++) );
          if( !((x&1) || (y&1)) ) { rgba2uv( *rgb, *(out_U++), *(out_V++) );
          }
        }
      }
    } else {
      rt_err( "can't copy img_t to YV12_buf, no (understood) pels data present in img_t" );
    }
  }


  struct asio_t {
    asio_t( void ) : frame_timer(io), quit_event(io), lb_event(io) { }
    io_service_t io;
    deadline_timer_t frame_timer;
    time_duration frame_dur;
    deadline_timer_t quit_event;
    lb_event_t lb_event;
  };
  
  // for now, our quit model is to stop all io on quit unless someone
  // has requested the quit event, in which case we assume they will
  // handle things.
  io_service_t & get_io( disp_win_t * const dw ) { return dw->asio->io; }
  deadline_timer_t & get_quit_event( disp_win_t * const dw ) {
    dw->stop_io_on_quit = 0;
    dw->asio->quit_event.expires_from_now( pos_infin ); // init event so it won't happen till we set it
    return dw->asio->quit_event;
  }
  lb_event_t & get_lb_event( disp_win_t * const dw ) {
    return dw->asio->lb_event;
  }

  void on_frame( disp_win_t * const dw, error_code const & ec ) {
    if( ec ) { return; } // handle?
    //printf( "dw->asio->frame_timer->expires_at()=%s\n", str(dw->asio->frame_timer.expires_at()).c_str() );
    dw->drain_sdl_events_and_redisplay();
    if( !dw->done ) { 
      dw->asio->frame_timer.expires_at( dw->asio->frame_timer.expires_at() + dw->asio->frame_dur );
      dw->asio->frame_timer.async_wait( bind( on_frame, dw, _1 ) ); 
    }
    else { 
      dw->asio->quit_event.cancel();
      SDL_Quit();
      if( dw->stop_io_on_quit ) { dw->asio->io.stop(); }
    }
  }

  disp_win_t::disp_win_t( void ) : cam_mode(1), zoom(0), stop_io_on_quit(1), asio( new asio_t ) { reset_cam(); }

  void disp_win_t::reset_cam( void ) {
    for( uint32_t i = 0; i != 3; ++i ) { cam_rot[i] = 0.0f; cam_pos[i] = 0.0f; }
    cam_pos[1] = 10.0f;
    cam_pos[2] = 10.0f;
    cam_rot[1] = 70.0f;
      
  }

  // FIXME: the size of imgs and the w/h of the img_t's inside imgs
  // may not change after setup, but this is not checked.

  void disp_win_t::disp_setup( p_img_t const & img ) {
    p_vect_p_img_t req_imgs( new vect_p_img_t );
    req_imgs->push_back( img );
    disp_setup( req_imgs );
  }

  void disp_win_t::disp_setup( vect_p_img_t const & imgs_ ) {
    p_vect_p_img_t req_imgs( new vect_p_img_t(imgs_) );
    disp_setup( req_imgs );
  }

  p_vect_p_img_t disp_win_t::disp_setup( vect_u32_pt_t const & disp_img_szs ) {
    p_vect_p_img_t req_imgs( new vect_p_img_t );
    for( vect_u32_pt_t::const_iterator i = disp_img_szs.begin(); i != disp_img_szs.end(); ++i ) {
      p_img_t img( new img_t );
      img->set_sz_and_alloc_pels( *i );
      img->fill_with_pel( grey_to_pel( 128 ) );
      req_imgs->push_back( img );
    }
    disp_setup( req_imgs );
    return req_imgs;
  }

  p_layout_elem_t make_p_layout_elem_t_init_and_check_unused_from_lexp( p_lexp_t const & lexp, nesi_init_arg_t * const nia );

  void disp_win_t::disp_setup( p_vect_p_img_t const & imgs_ ) {
    imgs = imgs_;
    img_annos.resize( imgs->size() );
    assert_st( !imgs->empty() );
    
    //uint32_t const pixel_format = SDL_PIXELFORMAT_ABGR8888;
    uint32_t const pixel_format = SDL_PIXELFORMAT_YV12;
    YV12_buf.reset( new YV12_buf_t );

    map_str_u32_pt_t img_szs;
    for( uint32_t i = 0; i != imgs->size(); ++i ) { must_insert( img_szs, str(i), imgs->at(i)->sz ); }
    map_str_u32_pt_t img_ncs;
    if( !disp_layout ) {
      string default_disp_layout_str = "(mode=horiz,kids=(";
      for( uint32_t i = 0; i != imgs->size(); ++i ) { default_disp_layout_str += string(i?",":"")+"_=(mode=leaf:"+str(i)+",kids=())"; }
      default_disp_layout_str += "))";
      disp_layout = make_p_layout_elem_t_init_and_check_unused_from_lexp( parse_lexp( default_disp_layout_str ), 0 );
    }
    disp_layout->calc_layout( img_szs, img_ncs );
    imgs_buf_nc.clear();
    for( uint32_t i = 0; i != imgs->size(); ++i ) { imgs_buf_nc.push_back( must_find( img_ncs, str(i) ) ); }
    u32_pt_t & img_sz = disp_layout->sz;
    // make full texture target w/h even for simplicity of YUV UV (2x downsampled) planes
    if( img_sz.d[0] & 1 ) { ++img_sz.d[0]; }
    if( img_sz.d[1] & 1 ) { ++img_sz.d[1]; }
    YV12_buf->set_sz_and_alloc( img_sz );

    if( SDL_Init( SDL_INIT_VIDEO ) < 0 ) { rt_err( strprintf( "Couldn't initialize SDL: %s\n", SDL_GetError() ) ); }

    if( window_sz == u32_pt_t() ) { window_sz = {YV12_buf->w,YV12_buf->h}; } // if no window size, use 'native' size of render-target-texture
    if( !window ) {
      window = make_p_SDL( SDL_CreateWindow( "boda display", 
                                             SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                             window_sz.d[0], window_sz.d[1],
                                             SDL_WINDOW_RESIZABLE) );
    }
    if( !window ) { rt_err( strprintf( "Couldn't set create window: %s\n", SDL_GetError() ) ); }
    if( !renderer ) {
      renderer = make_p_SDL( SDL_CreateRenderer( window.get(), -1, 0) ) ;
    }
    if (!renderer) { rt_err( strprintf( "Couldn't set create renderer: %s\n", SDL_GetError() ) ); }

#if 0
    SDL_RendererInfo  rinfo;
    SDL_GetRendererInfo( renderer.get(), &rinfo );
    printf( "rinfo.name=%s\n", str(rinfo.name).c_str() );
#endif

    tex.reset();
    tex = make_p_SDL( SDL_CreateTexture( renderer.get(), pixel_format, SDL_TEXTUREACCESS_STREAMING, 
                                         YV12_buf->w, YV12_buf->h ) );

    if( !tex ) { rt_err( strprintf( "Couldn't set create texture: %s\n", SDL_GetError()) ); }


    displayrect.reset( new SDL_Rect );
    displayrect->x = 0;
    displayrect->y = 0;
    displayrect->w = window_sz.d[0];
    displayrect->h = window_sz.d[1];


    paused = 0;
    done = 0;
    frame_cnt = 0;
    int const fps = 60;

    asio->frame_dur = microseconds( 1000 * 1000 / fps );
    asio->frame_timer.expires_from_now( time_duration() );
    asio->frame_timer.async_wait( bind( on_frame, this, _1 ) );
    asio->lb_event.expires_from_now( pos_infin ); // init event so it won't happen till we set it

    // font setup
    if( !font_setup_done.v ) {
      if( TTF_Init() < 0 ) { rt_err_sdl( "Couldn't initialize TTF" ); }

      string const font_fn = py_boda_dir() +"/fonts/DroidSansMono.ttf"; // FIXME: use boost filesystem?
      uint32_t const ptsize = 18;
      font.reset( TTF_OpenFont(font_fn.c_str(), ptsize), TTF_CloseFont );
      if( !font ) { rt_err_sdl( strprintf( "Couldn't load %s pt font from %s", str(ptsize).c_str(), font_fn.c_str() ).c_str() ); }

      int const renderstyle = TTF_STYLE_NORMAL;
      int const hinting = TTF_HINTING_MONO;
      int const kerning = 0;

      //printf( "TTF_FontFaceIsFixedWidth()=%s\n", str(TTF_FontFaceIsFixedWidth(font.get())).c_str() );
      TTF_SetFontStyle( font.get(), renderstyle );
      TTF_SetFontOutline( font.get(), 0 );
      TTF_SetFontKerning( font.get(), kerning );
      TTF_SetFontHinting( font.get(), hinting );
      font_setup_done.v = 1;
    }
  }

  // replace a backing image wholesale. note: image size must not change. note2: will not update display; call
  // update_disp_imgs() below for that. this function is an alternative to shared-ownership of the backing images, and
  // can also replace usages where shared_ref_to_backing_image->share_pels_from() was called.
  void disp_win_t::update_disp_img( uint32_t const pix, p_img_t const & new_img ) {
    assert_st( pix < imgs->size() );
    assert_st( imgs->at(pix)->sz == new_img->sz );
    imgs->at(pix) = new_img;
  }
    
  // call when changes to imgs should be reflected/copied onto the display texture
  void disp_win_t::update_disp_imgs( void ) {
    if( paused ) { return; }
    assert_st( imgs->size() == imgs_buf_nc.size() );
    for( uint32_t i = 0; i != imgs->size(); ++i ) {
      u32_pt_t const & img_nc = imgs_buf_nc[i];
      img_to_YV12( *YV12_buf, imgs->at(i), img_nc.d[0], img_nc.d[1] );
    }
    SDL_UpdateTexture( tex.get(), NULL, YV12_buf->d.get(), YV12_buf->w );
  }

  p_img_t disp_win_t::get_borrowed_output_frame( void ) { return YV12_buf->as_img_t_yuv_borrowed(); }
  
  void disp_win_t::update_img_annos( uint32_t const & img_ix, p_vect_anno_t const & annos ) { 
    if( paused ) { return; }
    img_annos.at(img_ix) = annos; 
  }

  void disp_win_t::on_lb( int32_t const x, int32_t const y ) { 
    //printf( "x=%s y=%s\n", str(x).c_str(), str(y).c_str() );
    i32_pt_t const xy{x,y};
    // FIXME: dup'd constants/code with annotation code below ...
    i32_pt_t const disp_sz{displayrect->w,displayrect->h}; // display window is of size displayrect w,h
    i32_pt_t const disp_off{displayrect->x,displayrect->y}; // display window x,y is the offset where the neg_corner of the texure will be drawn. 
    i32_pt_t const tex_sz{ int32_t(YV12_buf->w), int32_t(YV12_buf->h) }; // the texture is always it is always drawn resized to the window size (regardless of offset)
    uint32_t out_x = 0;
    asio->lb_event.img_ix = uint32_t_const_max; // default: not inside any image
    assert_st( imgs->size() == imgs_buf_nc.size() );
    for( uint32_t i = 0; i != imgs->size(); ++i ) { 
      p_img_t const & img = imgs->at(i);
      // calculate what region in the display window this image occupies
      // note: result may be clipped offscreen if it is outside of the visible area of {{0,0},disp_sz}
      i32_pt_t const img_nc = u32_to_i32( imgs_buf_nc[i] ); // { int32_t(out_x), int32_t(YV12_buf->h) - int32_t(img->sz.d[1]) };
      i32_pt_t const disp_img_nc = (img_nc*disp_sz/tex_sz) + disp_off;
      i32_pt_t const img_sz = u32_to_i32( img->sz );
      i32_pt_t const disp_img_sz = img_sz*disp_sz/tex_sz;
      i32_box_t const disp_img_box = {disp_img_nc,disp_img_nc+disp_img_sz};
      //printf( "disp_img_box=%s\n", str(disp_img_box).c_str() );
      //printf( "img_sz=%s\n", str(img_sz).c_str() );
      out_x += imgs->at(i)->sz.d[0];
      if( disp_img_box.contains( pel_to_box( i32_pt_t{x,y} ) ) ) {
	i32_pt_t const img_xy = (xy - disp_img_nc)*img_sz/disp_img_sz;
	//printf( "i=%s img_xy=%s\n", str(i).c_str(), str(img_xy).c_str() );
	assert_st( asio->lb_event.img_ix == uint32_t_const_max ); // shouldn't be inside multiple images
	asio->lb_event.img_ix = i;
	asio->lb_event.xy = img_xy;
      }
    }
    // FIXME/HACK: we can't seem to prevent this 'event' from firing for no-obvious-reason at startup. but, since we
    // init valid to 0 in the ctor, we can ignore such firings prior to getting to this line the first time.
    asio->lb_event.valid = 1; 
    asio->lb_event.cancel();
  }

  void disp_win_t::update_dr_for_window_and_zoom( u32_pt_t const & new_win_sz ) {
    i32_pt_t const orig_dr_sz = i32_pt_t{displayrect->w,displayrect->h};
    i32_pt_t const orig_dr_xy = i32_pt_t{displayrect->x,displayrect->y};
    i32_pt_t const new_dr_sz = u32_to_i32(new_win_sz).scale_and_round( pow( 1.3, zoom ) ); 
    displayrect->w = new_dr_sz.d[0];
    displayrect->h = new_dr_sz.d[1];
    // adjust dr nc to keep center point in center
    i32_pt_t const new_dr_xy = ( u32_to_i32(new_win_sz).scale_and_round(.5)*orig_dr_sz -
				 (u32_to_i32(window_sz).scale_and_round(.5)-orig_dr_xy)*new_dr_sz )/orig_dr_sz;
    displayrect->x = new_dr_xy.d[0];
    displayrect->y = new_dr_xy.d[1];
    window_sz = new_win_sz;
  }

  void disp_win_t::drain_sdl_events_and_redisplay( void ) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
      case SDL_WINDOWEVENT:
	if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
	  SDL_RenderSetViewport(renderer.get(), NULL);
	  update_dr_for_window_and_zoom(i32_to_u32(i32_pt_t{event.window.data1,event.window.data2}));
	}
	break;
      case SDL_MOUSEBUTTONDOWN:
        pan_pin = i32_pt_t{event.button.x,event.button.y};
        pan_orig_dr = i32_pt_t{displayrect->x,displayrect->y};
        pan_orig_cam_x = cam_pos[0];
        pan_orig_cam_y = cam_pos[1];
        pan_orig_cam_z = cam_pos[2];
        pan_orig_cam_rx = cam_rot[0];
        pan_orig_cam_ry = cam_rot[1];
        pan_orig_cam_rz = cam_rot[2];
        if( event.button.button == SDL_BUTTON_RIGHT ) { asio->lb_event.set_is_lb(); on_lb( event.button.x, event.button.y ); }
	break;
      case SDL_MOUSEMOTION:
        if( cam_mode == 0 ) { // 2D-adj
          if (event.motion.state&SDL_BUTTON(1)) {
            i32_pt_t const pan_to = pan_orig_dr + (i32_pt_t{event.motion.x,event.motion.y} - pan_pin);
            displayrect->x = pan_to.d[0];
            displayrect->y = pan_to.d[1];
          }
        } else { // 3D-adj
          if (event.motion.state&SDL_BUTTON(2)) {
            i32_pt_t const pan_to = i32_pt_t{event.motion.x,event.motion.y} - pan_pin;
            cam_pos[0] = pan_orig_cam_x - pan_to.d[0];
            cam_pos[2] = pan_orig_cam_z - pan_to.d[1];
          }
          if (event.motion.state&SDL_BUTTON(1)) {
            i32_pt_t const pan_to = i32_pt_t{event.motion.x,event.motion.y} - pan_pin;
            cam_rot[0] = pan_orig_cam_rx - pan_to.d[0];
            clamp_eq( cam_rot[0], -180.0f, 180.0f );
            cam_rot[1] = pan_orig_cam_ry + pan_to.d[1];
            clamp_eq( cam_rot[1], 0.0f, 90.0f );
          }
          if (event.motion.state&SDL_BUTTON(3)) {
            i32_pt_t const pan_to = i32_pt_t{event.motion.x,event.motion.y} - pan_pin;
            cam_pos[0] = pan_orig_cam_x - pan_to.d[0];
            cam_pos[1] = pan_orig_cam_y - pan_to.d[1];
          }

        }
	break;
      case SDL_MOUSEWHEEL:
        if( cam_mode == 0 ) { // 2D-adj
          zoom += event.wheel.y;
          min_eq( zoom,  10 );
          max_eq( zoom, -10 );
          update_dr_for_window_and_zoom( window_sz );
        } else {
          cam_rot[2] += - event.wheel.y;
        }
	break;
      case SDL_KEYDOWN:
	if( 1 ) { // generate/forward keydown event to disp_util parent/user/client
	  int mx,my; SDL_GetMouseState( &mx, &my );
	  asio->lb_event.set_is_key( event.key.keysym.sym ); on_lb( mx, my );
	}
	if( event.key.keysym.sym == SDLK_z ) {
          cam_mode ^= 1;
          printf( "cam_mode=%s\n", str(cam_mode).c_str() );
	  break;
	}
	if( event.key.keysym.sym == SDLK_s ) {
	  for( uint32_t i = 0; i != imgs->size(); ++i ) {
	    imgs->at(i)->save_fn_png( strprintf( "ss_%s.png", str(i).c_str() ) );
	  }
	  paused = 1;
	  break;
	}
	if( event.key.keysym.sym == SDLK_j ) {
	  for( uint32_t i = 0; i != imgs->size(); ++i ) {
	    imgs->at(i)->save_fn_jpeg( strprintf( "ss_%s.jpg", str(i).c_str() ) );
	  }
	  paused = 1;
	  break;
	}
	if( event.key.keysym.sym == SDLK_k ) {
          YV12_buf->as_img_t_yuv_borrowed()->save_fn_jpeg( strprintf( "ss.jpg" ) );
	  paused = 1;
	  break;
	}
	if (event.key.keysym.sym == SDLK_SPACE) {
	  paused = !paused;
	  break;
	}
	if (event.key.keysym.sym == SDLK_r) {
	  imgs->at(0)->fill_with_pel( grey_to_pel( frame_cnt % 256 ) );
	  break;
	}
	if (event.key.keysym.sym != SDLK_ESCAPE) {
	  break;
	}
      case SDL_QUIT:
	done = SDL_TRUE;
	break;
      }
    }
    sdl_set_color_from_pel( renderer, 0 );
    SDL_RenderClear( renderer.get() );
    SDL_RenderCopy( renderer.get(), tex.get(), NULL, displayrect.get() );

    i32_pt_t const disp_sz{displayrect->w,displayrect->h}; // display window is of size displayrect w,h
    i32_pt_t const disp_off{displayrect->x,displayrect->y}; // display window x,y is the offset where the neg_corner of the texure will be drawn. 
    i32_pt_t const tex_sz = { int32_t(YV12_buf->w), int32_t(YV12_buf->h) }; // the texture is always it is always drawn resized to the window size (regardless of offset)
    double_pt_t const tex_to_disp_scale{ double(disp_sz.d[0])/tex_sz.d[0], double(disp_sz.d[1])/tex_sz.d[1] };
    for( uint32_t i = 0; i != imgs->size(); ++i ) { 
      // calculate what region in the display window this image occupies
      // note: result may be clipped offscreen if it is outside of the visible area of {{0,0},disp_sz}
      i32_pt_t const img_nc = u32_to_i32( imgs_buf_nc.at(i) );
      // draw annotations
      p_vect_anno_t const & annos = img_annos.at(i);
      if( !annos ) { continue; }
      for( vect_anno_t::const_iterator i = annos->begin(); i != annos->end(); ++i ) {
	// render box
	sdl_set_color_from_pel( renderer, i->box_color );
	i32_box_t const anno_box = (i->box+img_nc).scale_and_round(tex_to_disp_scale) + disp_off;
	SDL_Rect sdl_anno_box = box_to_sdl( anno_box  );
	//printf( "anno_box=%s\n", str(anno_box).c_str() );
	if( i->fill ) { SDL_RenderFillRect(renderer.get(), &sdl_anno_box ); } 
	else { SDL_RenderDrawRect(renderer.get(), &sdl_anno_box ); }
	// render string
	if( i->str.empty() ) { continue; } 
	p_SDL_Texture str_tex;
	// note: we might use anno_box.w instead of a hard-coded 800
	// here (or maybe disp_img_sz.d[0]) as the auto-wrapping size
	// here to keep the text inside anno_box in X. but, that's not
	// really what we want: if we ever auto-wrap it's probably
	// bad, and if we ever can't fix a word on a line it's bad. so
	// we might just prefer to use a large/infinite value for the
	// wrapping (and let text overflow the anno_box in X as
	// needed). but, the wrapLength must be non-zero to enable
	// wrapping at all, and then it determines the width of the
	// returned surface. so ... we pick a hopefully okay-ish value
	// ...
        p_SDL_Surface text( TTF_RenderText_Blended_Wrapped( font.get(), i->str.c_str(), color_to_sdl(i->str_color), 800 ), 
			    SDL_FreeSurface );
	if( !text ) { printf("text render failed\n"); }
        else { 
	  //assert_st( text->w == anno_box.w );
	  SDL_Rect text_box = sdl_anno_box; // for - corner
	  text_box.h = text->h; // may be +- height of anno_box
	  text_box.w = text->w; // may be +- width of anno_box
	  p_SDL_Surface text_shadow( SDL_CreateRGBSurface(SDL_SWSURFACE, text->w, text->h, 32,
							  0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000), SDL_FreeSurface );
	  SDL_FillRect( text_shadow.get(), 0, rgba_to_pel( 0,0,0,120 ) );
	  str_tex = make_p_SDL( SDL_CreateTextureFromSurface( renderer.get(), text_shadow.get() ) ); 
	  SDL_RenderCopy( renderer.get(), str_tex.get(), 0, &text_box );
	  str_tex = make_p_SDL( SDL_CreateTextureFromSurface( renderer.get(), text.get() ) ); 
	  SDL_RenderCopy( renderer.get(), str_tex.get(), 0, &text_box );
	}

      }
    }      
    ++frame_cnt;
    SDL_RenderPresent( renderer.get() );
  }
#include"gen/disp_util.H.nesi_gen.cc"

}
