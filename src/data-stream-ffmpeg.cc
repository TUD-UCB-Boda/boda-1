// Copyright (c) 2017, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"has_main.H"
#include"str_util.H"
#include"data-stream.H"
#include"nesi.H"
extern "C" {
#include"libavformat/avformat.h"
#include"libavcodec/avcodec.h"
#include"libavutil/opt.h"
#include"libswscale/swscale.h"
}
#include"img_io.H"
#include<time.h>
#include"timers.H"

namespace boda 
{

  typedef shared_ptr< SwsContext > p_SwsContext;

  struct data_stream_ffmpeg_src_t : virtual public nesi, public data_stream_tagged_frames_t // NESI(
                                    // help="parse file with ffmpeg (libavformat,...) output one block per raw video frame",
                                    // bases=["data_stream_tagged_frames_t"], type_id="ffmpeg-src")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    
    filename_t fn; //NESI(req=1,help="input filename")
    uint32_t stream_index; //NESI(default="0",help="ffmpeg stream index from which to extract frames from")
    uint32_t output_rgba; //NESI(default="0",help="if non-zero, use sws_scale() to output 'normal' (not YUV) img_t's with RGBA data")
    uint32_t base_timestamp_from_fn; //NESI(default="0",help="if non-zero, use filename to get a base timestamp to add to output timestamps.")

    virtual string get_pos_info_str( void ) { return strprintf( "tot_num_read=%s", str(tot_num_read).c_str() ); }

    virtual bool seek_to_block( uint64_t const & frame_ix ) { return false; }

    AVFormatContext *ic;
    AVCodecContext *avctx;

    // the rational time_base for the output stream (for setting timestamps in gen_timestamp_from_fn mode)
    uint64_t time_base_num;
    uint64_t time_base_den;
    uint64_t base_ts;
    
    data_stream_ffmpeg_src_t( void ) : ic( 0 ) { }
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      base_ts = 0;
      if( base_timestamp_from_fn ) {
        string const time_fmt = "YYYYMMDD_HHMMSS";
        string const time_fmt_strptime = "%Y%m%d_%H%M%S";
        string const ext_str = ".avi";
        // assume fn ends in a time_fmt timestamp, and convert it to nanoseconds-since-epoch
        if( fn.exp.size() < (time_fmt.size() + ext_str.size()) ) {
          rt_err( strprintf( "fn.exp=%s doesn't end in a timestamp with format %s + .avi\n", str(fn.exp).c_str(), str(time_fmt).c_str() ) );
        }
        string const fn_ts_str = fn.exp.substr( fn.exp.size() - (time_fmt.size()+ext_str.size()), time_fmt.size() );
        tm fn_tm;
        char * const ret = strptime( fn_ts_str.c_str(), time_fmt_strptime.c_str(), &fn_tm );
        if( !ret ) {
          rt_err( strprintf( "strptime() failed to convert timestamp of %s into a tm struct.\n", str(fn_ts_str).c_str() ) );
        }
        time_t const fn_time = mktime( &fn_tm );
        if( fn_time == -1 ) {
          rt_err( strprintf( "mktime() failed to convert timestamp of %s into seconds-since-epoch.\n", str(fn_ts_str).c_str() ) );
        }
        base_ts = fn_time * secs_to_nsecs(1);
      }
      
      av_register_all(); // NOTE/FIXME: in general, this should be safe to call multiple times. but, there have been bugs wrt that ...
      ic = avformat_alloc_context();
      if (!ic) { rt_err( "avformat_alloc_context() failed" ); }
      //ic->interrupt_callback.callback = decode_interrupt_cb;
      //ic->interrupt_callback.opaque = is;
      AVDictionary * format_opts = NULL;
      // note: by default, ffplay sets "scan_all_pmts" to 1 here. but, perhaps we can ignore that, since it's unclear if
      // it's relevant to us -- seems only to be for mpegts containers, and then only sometimes relevant?
      string const ffmpeg_url = "file:" + fn.exp;
      //AVInputFormat *iformat;
      int err;
      err = avformat_open_input(&ic, ffmpeg_url.c_str(), NULL, &format_opts);
      if( err < 0 ) { rt_err( strprintf( "avformat_open_input failed for ffmpeg_url=%s\n", str(ffmpeg_url).c_str() ) ); }
      // note: here, we could check that all options were consumed. but, we're not setting any, so why bother. see the
      // relevant check in ffplay.c
      err = avformat_find_stream_info(ic, NULL);
      if( err < 0 ) { printf( "warning: avformat_find_stream_info() failed for ffmpeg_url=%s\n", str(ffmpeg_url).c_str() ); }

      if( !( stream_index < ic->nb_streams ) ) {
        rt_err( strprintf( "user requested (zero-based) stream_index=%s, but ffmpeg says there are only ic->nb_streams=%s streams.\n",
                           str(stream_index).c_str(), str(ic->nb_streams).c_str() ) );
      }
    
      for( uint32_t i = 0; i != ic->nb_streams; ++i ) {
        // FIXME: for no obvious reason, av_dump_format() seems to print nothing -- maybe an stdio/iostreams or other
        // C++-and-ffmpeg issue?
#if 0
        printf( "av_dump_format for stream: i=%s\n", str(i).c_str() );
        av_dump_format(ic, i, ffmpeg_url.c_str(), 0);
#endif
        ic->streams[i]->discard = ( i == stream_index ) ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
      }
      AVStream * const vid_st = ic->streams[stream_index];
      time_base_num = vid_st->time_base.num;
      time_base_den = vid_st->time_base.den;
      // FIXME/NOTE: it seems we could use either a direct check on vid_st_type or avformat_match_stream_specifier here. hmm.
      // AVMediaType vid_st_type = vid_st->codecpar->codex_type;
      int const avmss_ret = avformat_match_stream_specifier( ic, vid_st, "v" );
      assert_st( avmss_ret >= 0 );
      if( avmss_ret == 0 ) { rt_err( strprintf( "stream stream_index=%s is not a video stream", str(stream_index).c_str() ) ); }
      init_video_stream_decode( vid_st );

      // FIXME: need to close input (which calls avformat_free_context() internally)
      // avformat_close_input(&ic);

    }


    void init_video_stream_decode( AVStream * const vid_st ) {
      if( out_dims ) { return; } // for raw mode, no decoding will be done, so don't init codec (we might not be able to anyway)
      avctx = avcodec_alloc_context3(NULL);
      if (!avctx) { rt_err( "avcodec_alloc_context3() failed" ); }

      int avcodec_ret;
#if FFMPEG_31
      avcodec_ret = avcodec_parameters_to_context(avctx, vid_st->codecpar);
      if( avcodec_ret < 0 ) { rt_err( "avcodec_parameters_to_context() failed" ); }
#else
      avctx = vid_st->codec;
#endif
      av_codec_set_pkt_timebase(avctx, vid_st->time_base);

      AVCodec *codec;
      codec = avcodec_find_decoder(avctx->codec_id);

      if( !codec ) { rt_err( strprintf( "no codec could be found for id avctx->codex_id=%s\n", str(avctx->codec_id).c_str() ) ); }
      avctx->codec_id = codec->id;
      
      AVDictionary *opts = NULL;

      // this seems nice to set ... but what happens if we don't have it? for now, we die/fail.
      if(codec->capabilities & AV_CODEC_CAP_DR1) {
        avctx->flags |= CODEC_FLAG_EMU_EDGE;
      }
      else {
        rt_err( "maybe-unsupported/FIXME: codec without AV_CODEC_CAP_DR1" );
      }

      if (!av_dict_get(opts, "threads", NULL, 0)) {
        av_dict_set(&opts, "threads", "auto", 0);
      }
      
      av_dict_set(&opts, "refcounted_frames", "1", 0);
      avcodec_ret = avcodec_open2(avctx, codec, &opts);
      if( avcodec_ret < 0 ) { rt_err( "avcodec_open2() failed" ); }

      // check for any unconsume (unrecognized) options
      AVDictionaryEntry *t = NULL;
      if ((t = av_dict_get(opts, "", NULL, AV_DICT_IGNORE_SUFFIX))) {
        rt_err( strprintf( "unknown code option '%s'\n", t->key ) );
      }
      
    }

    // since we can't seem to fish into sws_ctx, we need to cache the params we used to create it so out lazy init of
    // the ctx can work. i wish we knew if lazy init was really needed/usefull, but i guess now that it's done it's not
    // worth investigating/takeing-back-out?
    p_SwsContext sws_ctx;
    uint32_t sws_w, sws_h;
    AVPixelFormat sws_sf, sws_df;
    
    void ensure_sws_ctx( uint32_t const & width, uint32_t const & height,
                         AVPixelFormat const & src_fmt, AVPixelFormat const & dst_fmt )
    {
      // assumes dest w/h same as src, since currently we don't ever scale
      if( sws_ctx ) {
        if( (sws_w == width) && (sws_h == height) &&
            (sws_sf == src_fmt) && (sws_df == dst_fmt) ) { return; }
        sws_ctx.reset();
      }
      assert_st( !sws_ctx );
      sws_ctx.reset( sws_getContext( width, height, src_fmt, width, height, dst_fmt, SWS_POINT, 0, 0, 0 ), sws_freeContext );
      sws_w = width; sws_h = height; sws_sf = src_fmt; sws_df = dst_fmt;
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      assert_st( ic );
      data_block_t ret = db;
      // set compatibility defaults for meta and tag, for now. can be overridded by tag/mega options. tag name is
      // obvs. not great ...
      ret.meta = "image";
      ret.tag = "camera-dumpvideo";
      AVPacket pkt;
      while( 1 ) {
        int const err = av_read_frame(ic, &pkt);
        if( err < 0 ) { return ret; }
        assert_st( (uint32_t)pkt.stream_index == stream_index ); // AVDISCARD_ALL setting for other streams in init() should guarentee this
        ret.timestamp_ns = base_ts + ( pkt.pts * time_base_num * secs_to_nsecs(1) / time_base_den ); // use packet pts as ts

        if( out_dims ) { // raw mode
          ret.nda = make_shared<nda_t>( dims_t{ vect_uint32_t{uint32_t(pkt.size)}, vect_string{ "v" }, "uint8_t" } );
          std::copy( pkt.data, pkt.data + pkt.size, (uint8_t *)ret.d() );
          assert_st( ret.nda );
          data_stream_block_done_hook( ret );
          return ret;
        }

        int got_frame = 0;
        AVFrame *frame = av_frame_alloc();
        int const decode_ret = avcodec_decode_video2(avctx, frame, &got_frame, &pkt);
        if( decode_ret < 0 ) { rt_err("avcodec_decode_video2() failed"); }
        if( decode_ret != pkt.size ) { rt_err("decode didn't consume entire packet?"); }

        if( got_frame ) {
          if( frame->format != AV_PIX_FMT_YUV420P ) {
            rt_err( "only the AV_PIX_FMT_YUV420P pixel format is currently supported for decoded output (adding conversions is TODO)" );
          }
          ret.as_img = make_shared< img_t >();
          // for now, we still require the frames to come out of the decoder as AV_PIX_FMT_YUV420P, but we will optionall
          // convert them to RGBA. 
          if( output_rgba ) {
            ret.as_img->set_sz_and_alloc_pels( {(uint32_t)frame->width,(uint32_t)frame->height} );
            ensure_sws_ctx( frame->width, frame->height, (AVPixelFormat)frame->format, AV_PIX_FMT_RGBA );
            uint8_t * dest_planes[3] = {ret.as_img->pels.get(),NULL,NULL};
            int dest_stride[3] = {frame->width*4,0,0};
            sws_scale( sws_ctx.get(), frame->data, frame->linesize, 0, frame->height, dest_planes, dest_stride); 
          }
          else {
            if( (frame->width&1) || (frame->height&1) ) {
              rt_err( strprintf( "only even frame sizes are supported, but frame->width=%s frame->height=%s\n",
                                 str(frame->width).c_str(), str(frame->height).c_str() ) );
            }
            // convert YUV planes to data block
            vect_p_nda_uint8_t yuv_ndas;
            ret.subblocks = make_shared<vect_data_block_t>();      
            for( uint32_t pix = 0; pix != 3; ++pix ) {
              uint32_t const subsample = pix ? 2 : 1;
              string const meta = string("YUV_") + string("YUV")[pix];
              uint32_t const ph = frame->height/subsample;
              uint32_t const pw = frame->width/subsample;
              p_nda_uint8_t yuv_nda = make_shared<nda_uint8_t>( dims_t{ vect_uint32_t{uint32_t(ph), uint32_t(pw)}, vect_string{ "y","x" },"uint8_t" });
              // fill in y,u, or v data
              for( uint32_t y = 0; y != ph; ++y ) {
                uint8_t * rb = frame->data[pix] + frame->linesize[pix]*y;
                std::copy( rb, rb + pw, &yuv_nda->at1(y) );
              }
              yuv_ndas.push_back( yuv_nda );
              if( pix == 0 ) {
                // ret.meta = meta; // oops, need to leave this as 'image' ... but we're not using the YUV meta stuff currently so ... barf.
                ret.nda = yuv_nda;
              } else {
                data_block_t uv_db;
                uv_db.meta = meta;
                uv_db.nda = yuv_nda;
                ret.subblocks->push_back( uv_db );
              }
            }
            ret.as_img->set_sz_and_pels_from_yuv_420_planes( yuv_ndas );
          }
          
          data_stream_block_done_hook( ret );
        }
        av_frame_free(&frame);
        if( got_frame ) { return ret; }
      }
    }
    
    // FIXME: it's not clear if this is reliably called at a 'good' time, but it's probably okay to aspirationally have
    // any teardown/dealloc code here (even if some is disabled/questionable) for ... safe keeping?
    ~data_stream_ffmpeg_src_t( void ) {
      // FIXME: add other needed ffmpeg free calls here?
      
      // FIXME: use shared_ptr deleter (or whatever) to dealloc these?
      // avcodec_free_context(&avctx); // only if FFMPEG_31
      // av_dict_free(&opts);

    }
  };


  struct data_stream_ffmpeg_sink_t : virtual public nesi, public data_stream_t // NESI(
                                    // help="read frames and output video file with ffmpeg (libavformat,libavcodec...)",
                                    // bases=["data_stream_t"], type_id="ffmpeg-sink")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    
    filename_t fn; //NESI(req=1,help="output filename")
    string out_fmt_name; //NESI(default="avi",help="output container ffmpeg-short-name")
    string codec_name; //NESI(default="mpeg4",help="ffmpeg codec to use. note that, for the default of mpeg4, it seems that the fourcc will be FMP4")
    uint32_t encode_fps; //NESI(default="15",help="output video frames-per-second")
    uint64_t tot_num_read; // num blocks read so far
    uint32_t libx264_crf; //NESI(default="20",help="if codec is libx264, use this crf. 20->high quality, 30->ok, smaller")
    uint32_t bitrate; //NESI(default="6000000",help="bitrate. note: if codec is libx264, will be set to 0 and crf used instead")
    uint32_t raw_mode; //NESI(default=0,help="if set, don't open codec, and dump raw image data to packets")
    string raw_codec_tag; //NESI(default="XRAW",help="fourcc to emit in raw (no codec) mode")
    
    virtual string get_pos_info_str( void ) { return strprintf( "data_stream_ffmpeg_sink: tot_num_read=%s",
                                                                str(tot_num_read).c_str() ); }

    virtual bool seek_to_block( uint64_t const & frame_ix ) { return false; }

    AVFormatContext *ofc;
    AVCodecContext *octx;
    AVStream *ostr;
    AVCodec *codec;
    AVFrame *frame;

    data_stream_ffmpeg_sink_t( void ) { }
    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      ofc = 0;
      octx = 0;
      ostr = 0;
      codec = 0;
      frame = 0;
      tot_num_read = 0;
      // we defer the 'real' init till we get the first frame, so we have info on the desired output video size
    }
    void lazy_init( data_block_t const &db ) {
      int err;

      assert_st( !ofc ); // call only once.
      assert_st( db.as_img );
      av_register_all(); // NOTE/FIXME: in general, this should be safe to call multiple times. but, there have been bugs wrt that ...

      if( !raw_mode ) {
        // setup codec and codec context
        codec = avcodec_find_encoder_by_name( codec_name.c_str() );
        if( !codec ) { rt_err( strprintf( "avcodec_find_encoder_by_name() for codec=%s failed", str(codec_name).c_str() ) ); }
      }
      
      // set up format and format context
      AVOutputFormat * const out_fmt = av_guess_format( out_fmt_name.c_str(), 0, 0 );
      if( !out_fmt ) { rt_err( strprintf( "av_guess_format() for out_fmt_name=%s failed", str(out_fmt_name).c_str() ) ); }
      err = avformat_alloc_output_context2( &ofc, out_fmt, 0, 0 );
      if( err || (!ofc) ) { rt_err( "avformat_alloc_output_context2() failed" ); }
      // note: codec may be null below, if we're in raw mode
      ostr = avformat_new_stream( ofc, codec ); // note: use codec here. this is will setup the context inside the stream for this codec
      if( !ostr ) { rt_err( "av_new_stream() failed" ); }
      assert_st( ostr->id == (int)ofc->nb_streams - 1 ); // FIXME/NOTE: is this really not set properly by avformat_new_stream?if not, set.
      // octx = avcodec_alloc_context3(codec); // since we use the codec context inside the stream, we don't alloc/setup here ...
      // if( !octx ) { rt_err( "avcodec_alloc_context3() failed" ); }
      octx = ostr->codec; // ... we just use the one inside the stream.
      assert_st( octx );
      //av_dict_set( &opts, "vprofile", "baseline", 0 )

      octx->bit_rate = bitrate; // set from preset?
      octx->gop_size = 12; // set from preset?
      p_img_t const & fi = db.as_img; // first image, use to set sizes
      octx->width = fi->sz.d[0];  
      octx->height = fi->sz.d[1];
      octx->time_base.den = encode_fps;  
      octx->time_base.num = 1;  
      octx->pix_fmt = PIX_FMT_YUV420P;

      ostr->time_base = octx->time_base; // seems we need to copy this into the stream?
      if( ofc->oformat->flags & AVFMT_GLOBALHEADER ) { octx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; } // BLACK MAGIC!

      if( codec_name == "libx264" ) { set_libx264_medium_presets( octx ); }
      if( codec_name == "mpeg2video" ) { set_mpeg2video_presets( octx ); }
      if( raw_mode ) {
        octx->codec_type = AVMEDIA_TYPE_VIDEO;
        assert_st( raw_codec_tag.size() == 4 );
        octx->codec_tag = *(uint32_t *)raw_codec_tag.c_str();
      }
//      AVDictionary *opts = NULL;
//      av_dict_set( &o, "preset", "ultrafast", 0 );
      
      //av_opt_set( octx->priv_data, "preset", "ultrafast", 0); // does this do anything? how to check?
      if( !raw_mode ) {
        err = avcodec_open2( octx, codec, 0 );
        if( err ) { rt_err( "avcodec_open2() failed" ); }

        frame = av_frame_alloc();
        if( !frame ) { rt_err( "av_frame_alloc() failed" ); }

        frame->format = octx->pix_fmt;
        frame->width  = octx->width;
        frame->height = octx->height;
      }
      // for reference, we need to fill the fields of frame similarly to how av_image_alloc would:
      //ret = av_image_alloc(frame->data, frame->linesize, c->width, c->height,  c->pix_fmt, 32);


      err = avio_open( &ofc->pb, fn.exp.c_str(), AVIO_FLAG_WRITE );
      if( err ) { rt_err( strprintf("avio_open(): failed for file '%s' (expanded: '%s')",fn.in.c_str(), fn.exp.c_str() ) ); }
      printf( "fn.exp=%s\n", str(fn.exp).c_str() );
      avformat_write_header( ofc, 0 );
      
#if 0
      // check for any unconsumed (unrecognized) options
      AVDictionaryEntry *t = NULL;
      if ((t = av_dict_get(opts, "", NULL, AV_DICT_IGNORE_SUFFIX))) {
        rt_err( strprintf( "unknown codec option '%s'\n", t->key ) );
      }
      av_dict_free( &opts );
#endif
    }
    
    AVPacket pkt;
    virtual data_block_t proc_block( data_block_t const & db ) {
      p_img_t const & img = db.as_img;
      if( !db.valid() && !img ) { data_stream_deinit(); return db; } // treat null db as deinit/end-of-video. any more packets
      if( !img ) { rt_err( "ffmpeg-sink: expected a data block with an image."); } // FIXME: can't happen now?
      if( !ofc ) { lazy_init( db ); }
      assert_st( ofc && octx && ostr );

      av_init_packet( &pkt );
      pkt.data = NULL;    // packet data will be allocated by the encoder
      pkt.size = 0;
      pkt.stream_index = ostr->id;

      if( !raw_mode ) {
        assert_st( frame );
        assert_st( codec );
        frame->pts = tot_num_read;
        if( img->yuv_pels.size() != 3 ) { rt_err( "ffmpeg-sink: expected data block image to have 3 yuv_pels vectors."); }
        uint32_t const align_check_bytes = 32;
        assert_st( ! (frame->height & 1) );
        assert_st( ! (frame->width & 1) );
        assert_st( (uint32_t)frame->height == img->sz.d[1] );
        assert_st( (uint32_t)frame->width == img->sz.d[0] );
        // put pointers from yuv_ndas into frame
        for( uint32_t pix = 0; pix != 3; ++pix ) {
          uint32_t const subsample = pix ? 2 : 1;
          uint32_t const pw = frame->width/subsample;
          frame->linesize[pix] = pw;
          frame->data[pix] = img->yuv_pels.at(pix).get(); 
          // it's not clear what alignment we really need to guarentee for the planes of frame, if any. but the ffmpeg
          // encode example uses 32-byte alignment, so we test for that ...
          if( ((uintptr_t)frame->data[pix] % align_check_bytes) != 0 ) {
            rt_err( "unaligned pointer in YUV->ffmpeg conversion. FIXME/TODO: make source of YUV buf be aligned to 32 per plane." );
          }
        }

        int got_output = 0;
        int err = 0;
        err = avcodec_encode_video2( octx, &pkt, frame, &got_output );
        if( err ) { rt_err( "avcodec_encode_video2() failed" ); }

        if (got_output) {
          printf("Write frame %s (size=%5d)\n", str(tot_num_read).c_str(), pkt.size);
          err = av_interleaved_write_frame( ofc, &pkt );
          if( err ) { rt_err( "av_interleaved_write_frame() failed" ); }
          // av_free_packet(&pkt); // don't free, since consumed by write interleaved? or 'cause on stack? hmm
        }
      } else { // in raw mode, just dump raw image to packet
        // FIXME/TODO: allow downsample optionally here? add downsample data_stream_t?
        if( !img->pels.get() ) { 
          rt_err( "TODO: ffmpeg raw image writing only supports incoming rgba images (not YUV)" );
        }
        pkt.size = img->sz_raw_bytes();
        pkt.data = (uint8_t *)img->get_row_addr(0);
        pkt.pts = tot_num_read;
        pkt.dts = 0;
        pkt.flags = 0;
        pkt.side_data = 0;
        pkt.side_data_elems = 0;
        pkt.duration = 0;
        pkt.pos = -1;
        int const err = av_interleaved_write_frame( ofc, &pkt );
        if( err ) { rt_err( "av_interleaved_write_frame() failed" ); }
      }
      
      ++tot_num_read;
      return db;

    }

    void data_stream_deinit( void ) {
      if( !ofc ) { return; }
      assert_st( ofc && octx && ostr && codec && frame );
      av_write_trailer( ofc );

      avcodec_close( octx );
      av_frame_free( &frame );

      if( !(ofc->oformat->flags & AVFMT_NOFILE) && ofc->pb ) { avio_closep( &ofc->pb ); }
      avformat_free_context( ofc ); // frees ostr internally?

      // FIXME: free:  ostr, codec ?
      data_stream_init( 0 ); // happens to zero-out/clear all data, making object ready for another lazy_init
    }
    void set_libx264_medium_presets( AVCodecContext * const ctx ) {
      // from:  https://stackoverflow.com/questions/3553003/how-to-encode-h-264-with-libavcodec-x264
      // libx264-medium.ffpreset preset
      ctx->coder_type = 1;  // coder = 1
      ctx->flags|=CODEC_FLAG_LOOP_FILTER;   // flags=+loop
      ctx->me_cmp|= 1;  // cmp=+chroma, where CHROMA = 1
      //ctx->partitions|=X264_PART_I8X8+X264_PART_I4X4+X264_PART_P8X8+X264_PART_B8X8; // partitions=+parti8x8+parti4x4+partp8x8+partb8x8
      //ctx->me_method=ME_HEX;    // me_method=hex
      ctx->me_subpel_quality = 7;   // subq=7
      ctx->me_range = 16;   // me_range=16
      ctx->gop_size = 250;  // g=250
      ctx->keyint_min = 25; // keyint_min=25
      ctx->scenechange_threshold = 40;  // sc_threshold=40
      ctx->i_quant_factor = 0.71; // i_qfactor=0.71
      ctx->b_frame_strategy = 1;  // b_strategy=1
      ctx->qcompress = 0.6; // qcomp=0.6
      ctx->qmin = 10;   // qmin=10
      ctx->qmax = 51;   // qmax=51
      ctx->max_qdiff = 4;   // qdiff=4
      ctx->max_b_frames = 3;    // bf=3
      ctx->refs = 3;    // refs=3
      //ctx->directpred = 1;  // directpred=1
      ctx->trellis = 1; // trellis=1
      //ctx->flags2|=CODEC_FLAG2_BPYRAMID+CODEC_FLAG2_MIXED_REFS+CODEC_FLAG2_WPRED+CODEC_FLAG2_8X8DCT+CODEC_FLAG2_FASTPSKIP;  // flags2=+bpyramid+mixed_refs+wpred+dct8x8+fastpskip
      //ctx->weighted_p_pred = 2; // wpredp=2

// libx264-main.ffpreset preset
      //ctx->flags2|=CODEC_FLAG2_8X8DCT;c->flags2^=CODEC_FLAG2_8X8DCT;    // flags2=-dct8x8
      octx->bit_rate = 0; // variable bitrate
      av_opt_set(ctx->priv_data, "crf", str(libx264_crf).c_str(), AV_OPT_SEARCH_CHILDREN);
    }

    void set_mpeg2video_presets( AVCodecContext * const ctx ) {
      ctx->rc_max_rate = ctx->rc_min_rate = ctx->bit_rate;
      ctx->flags = CODEC_FLAG_INTERLACED_DCT;
      //ctx->flags2 = CODEC_FLAG2_INTRA_VLC | CODEC_FLAG2_NON_LINEAR_QUANT;
      ctx->qmin = 1;
      ctx->qmax = 1;
      ctx->rc_buffer_size = ctx->rc_initial_buffer_occupancy = 2000000;
      //ctx->rc_buffer_aggressivity = 0.25;
      ctx->profile = 0;
      ctx->level = 5;
      ctx->gop_size = 12;
      ctx->max_b_frames = 2;
    }    
  };

  
#include"gen/data-stream-ffmpeg.cc.nesi_gen.cc"

}
