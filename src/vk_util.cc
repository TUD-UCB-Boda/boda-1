// XXX include ordering important?
#include"boda_tu_base.H"
#include"rtc_compute.H"
#include "str_util.H"
#include "timers.H"
#include "vulkan.h"
#include <iostream>
#include <shaderc/shaderc.hpp>

#define DEBUG

// XXX needs massive refactoring (at least de-duplication, whitespace-errors and matching boda's naming convention)
// XXX remove several narrowing conversions and other warnings

namespace boda {
  // XXX improve/implement error handling
#define BAIL_ON_BAD_RESULT(result) \
  if (VK_SUCCESS != (result)) { fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(-1); }


  string vk_base_decls = R"rstr(
#version 450
#define GLSL
// XXX no typedef in GLSL
#define uint32_t uint
#define int32_t int
// XXX no native support for char/uint8_t
#define uint8_t uint

#define CUCL_BACKEND_IX 3
const uint32_t U32_MAX = 0xffffffff;
// XXX figure this out later
//typedef long long int64_t;
#define CUCL_GLOBAL_KERNEL
#define CUCL_DEVICE 
#define GASQ 
#define GLOB_ID_1D (gl_GlobalInvocationID.x)
#define LOC_ID_1D (gl_LocalInvocationID.x)
#define GRP_ID_1D (gl_WorkGroupID.x)
#define LOC_SZ_1D (gl_WorkGroupSize.x)
// shared variables are defined outside of the function
#define LOCSHAR_MEM
#define LSMASQ
// XXX check whether we need both barriers
#define BARRIER_SYNC groupMemoryBarrier(); barrier();

// note: it seems GLSL doesn't provide powf(), but instead overloads pow() for double and float. 
// so, we use this as a compatibility wrapper.
#define powf(v,e) pow(v,e)
// XXX figure this out later
//#define store_float_to_rp_half( val, ix, p ) vstore_half( val, ix, p )
#define store_float_to_rp_float( val, ix, p ) p[ix] = val

#define START_ARG 
#define END_ARG ;
#define FLOAT_CAST 

)rstr";

  struct vk_func_info_t {
    rtc_func_info_t info;
    VkShaderModule kern;
  };

  typedef map< string, vk_func_info_t > map_str_vk_func_info_t;
  typedef shared_ptr< map_str_vk_func_info_t > p_map_str_vk_func_info_t;

  // XXX find better name
  struct vk_buffer_info_t {
    VkBuffer buf;
    VkDeviceMemory mem;
  };

  typedef vector<vk_buffer_info_t> vect_vk_buffer_info_t;

  struct vk_var_info_t {
    vk_buffer_info_t bo;
    dims_t dims;
  };

  typedef map < string, vk_var_info_t > map_str_vk_var_info_t;
  typedef shared_ptr< map_str_vk_var_info_t > p_map_str_vk_var_info_t;

  typedef vector< VkQueryPool > vect_vk_query_pool_t;

  struct vk_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="Vulkan based rtc support",
			// bases=["rtc_compute_t"], type_id="vk" )
  {
  
    VkInstance instance;
    
    VkDebugReportCallbackEXT debug_report_callback;
    
    VkDevice device;

    VkPhysicalDeviceMemoryProperties mem_properties;

    VkQueue queue;

    uint32_t queue_family_index;

    VkCommandPool command_pool;
    zi_bool init_done;

    p_map_str_vk_var_info_t vis;
    p_map_str_vk_func_info_t kerns;

    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

  
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
							VkDebugReportFlagsEXT flags,
							VkDebugReportObjectTypeEXT objType,
							uint64_t obj,
							size_t location,
							int32_t code,
							const char* layer_prefix,
							const char* msg,
							void* user_data) {

      std::cerr << "validation layer: " << msg << std::endl;

      return VK_FALSE;
    }
    
    void init( void ) {
    
      const VkApplicationInfo application_info = {
	VK_STRUCTURE_TYPE_APPLICATION_INFO,
	0,
	"boda",
	0,
	"",
	0,
	VK_MAKE_VERSION(1, 0, 9) // XXX check
      };

      vector<const char*> layers;
      vector<const char*> extensions;

#ifdef DEBUG
      layers.push_back("VK_LAYER_LUNARG_standard_validation");
      extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#endif
      // XXX maybe use device extensions and layers
      const VkInstanceCreateInfo instance_create_info = {
	VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	0,
	0,
	&application_info,
	(uint32_t) layers.size(),
	layers.data(),
	(uint32_t) extensions.size(),
	extensions.data(),
      };

      BAIL_ON_BAD_RESULT(vkCreateInstance(&instance_create_info, 0, &instance));

#ifdef DEBUG
      const VkDebugReportCallbackCreateInfoEXT debug_callback_create_info = {
	VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
	0,
	VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
	debug_callback,
	0};

      // We have to explicitly load this function.
      auto vk_create_debug_report_callback_EXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
      BAIL_ON_BAD_RESULT(vk_create_debug_report_callback_EXT == nullptr ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS);

      // Create and register callback.
      BAIL_ON_BAD_RESULT(vk_create_debug_report_callback_EXT(instance, &debug_callback_create_info, NULL, &debug_report_callback));
#endif
    
      uint32_t physical_device_count = 0;
      BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physical_device_count, 0));

      VkPhysicalDevice* const physical_devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * physical_device_count);

      BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices));

      BAIL_ON_BAD_RESULT(physical_device_count ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED);

      uint32_t queue_family_count;

      // XXX just use device 0 for now

      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[0], &queue_family_count, 0);
      std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);

      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[0], &queue_family_count, queue_families.data());
      queue_family_index = 0;
      bool found_queue_family = false;
      for (; queue_family_index < queue_family_count; queue_family_index++) {
	VkQueueFamilyProperties props = queue_families[queue_family_index];
      
	if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
	  found_queue_family = true;
	  break;
	}
      }

      BAIL_ON_BAD_RESULT(found_queue_family ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED);

      // XXX investigate
      const float queue_priority = 1.0f;

    
      const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
	VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	0,
	0,
	queue_family_index,
	1,
	&queue_priority
      };

      // XXX device extensions and layers
      const VkDeviceCreateInfo device_create_info = {
	VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	0,
	0,
	1,
	&deviceQueueCreateInfo,
	0,
	0,
	0,
	0,
	0
      };

      BAIL_ON_BAD_RESULT(vkCreateDevice(physical_devices[0], &device_create_info, 0, &device));

      vkGetPhysicalDeviceMemoryProperties(physical_devices[0], &mem_properties);
  
      vkGetDeviceQueue(device, queue_family_index, 0, &queue);
    

      VkCommandPoolCreateInfo command_pool_create_info = {
	VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	0,
	0,
	queue_family_index
      };

      BAIL_ON_BAD_RESULT(vkCreateCommandPool(device, &command_pool_create_info, 0, &command_pool));

      init_done.v = true;
    }

    virtual string get_plat_tag( void ) {
      // XXX improve
      return "vk";
    }


    void compile( vect_rtc_func_info_t const & func_infos, rtc_compile_opts_t const & opts) {
      assert (init_done.v);
      timer_t t("vk compile");
      if (func_infos.empty()) return;
    
      for( vect_rtc_func_info_t::const_iterator i = func_infos.begin(); i != func_infos.end(); ++i ) {
	string src = vk_base_decls + get_rtc_base_decls() + i->func_src;
	
	if( gen_src ) {
	  ensure_is_dir( gen_src_output_dir.exp, 1 );
	  p_ostream out = ofs_open( strprintf( "%s/%s_%d.glsl", gen_src_output_dir.exp.c_str(), i->func_name.c_str(), (int) (i - func_infos.begin())));
	  (*out) << src << std::flush;
	}
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;

	options.SetOptimizationLevel(shaderc_optimization_level_size);
	shaderc::SpvCompilationResult module =
	  compiler.CompileGlslToSpv(src, shaderc_glsl_compute_shader, i->func_name.c_str(), options);
    
	std::vector<uint32_t> buffer;
	buffer = {module.cbegin(), module.cend()};

	if (buffer.size() == 0)
	  BAIL_ON_BAD_RESULT(VK_INCOMPLETE);
    
	VkShaderModuleCreateInfo shader_module_create_info = {
	  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	  0,
	  0,
	  buffer.size() * sizeof(uint32_t),
	  buffer.data()
	};

	VkShaderModule shader_module;
	BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shader_module_create_info, 0, &shader_module));

	must_insert(*kerns, i->func_name, vk_func_info_t{*i, shader_module});
      }      

    
    }
    
    // buf and mem of return value must be destroyed/freed
    vk_buffer_info_t create_buffer_info(VkDeviceSize size, VkBufferUsageFlags buffer_flags, VkMemoryPropertyFlags memory_flags) {

      const VkBufferCreateInfo buffer_create_info = {
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	0,
	0,
	size,
	buffer_flags,
	VK_SHARING_MODE_EXCLUSIVE,
	1,
	&queue_family_index
      };

    
      VkBuffer buf;
      BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &buffer_create_info, 0, &buf));

      
      // set memory_type_index to an invalid entry in the properties.memoryTypes array
      uint32_t memory_type_index = VK_MAX_MEMORY_TYPES;

      VkMemoryRequirements mem_reqs;
      vkGetBufferMemoryRequirements(device, buf, &mem_reqs);

      for (uint32_t k = 0; k < mem_properties.memoryTypeCount; k++) {
	if ((memory_flags & mem_properties.memoryTypes[k].propertyFlags) == memory_flags &&
	    (mem_reqs.memoryTypeBits & (1 << k))) {
	  memory_type_index = k;
	  break;
	}
      }

      BAIL_ON_BAD_RESULT(memory_type_index == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

      const VkMemoryAllocateInfo memory_allocate_info = {
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	0,
	mem_reqs.size,
	memory_type_index
      };
      VkDeviceMemory dev_mem;
      BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memory_allocate_info, 0, &dev_mem));
 
      BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, buf, dev_mem, 0));
     
      return {buf, dev_mem};
    }

    void buffer_copy(VkBuffer dst, VkBuffer src, VkDeviceSize size) {

      VkCommandBufferAllocateInfo command_buffer_allocate_info = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	0,
	command_pool,
	VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	1
      };

      VkCommandBuffer command_buffer;
      BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &command_buffer));

      VkCommandBufferBeginInfo command_buffer_begin_info = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	0,
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	0
      };

      BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info));

      VkBufferCopy copy = {0, 0, size};
      vkCmdCopyBuffer(command_buffer, src, dst, 1, &copy);


      BAIL_ON_BAD_RESULT(vkEndCommandBuffer(command_buffer));


      VkSubmitInfo submit_info = {
	VK_STRUCTURE_TYPE_SUBMIT_INFO,
	0,
	0,
	0,
	0,
	1,
	&command_buffer,
	0,
	0
      };
      BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submit_info, 0));
      BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));
    }

    
    void copy_nda_to_var (string const &vn, p_nda_t const & nda) {

      // XXX maybe we should use some kind of heuristic to avoid allocating
      // a new staging buffer everytime - var_to_nda as well
      vk_var_info_t const & vi = must_find( *vis, vn );
      assert( nda->dims == vi.dims);
      vk_buffer_info_t staging = create_buffer_info(nda->dims.bytes_sz(),
						    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      
      void *dev_ptr;
      BAIL_ON_BAD_RESULT(vkMapMemory(device, staging.mem, 0, nda->dims.bytes_sz(), 0, (void **)&dev_ptr));
      memcpy(dev_ptr, nda->rp_elems(), nda->dims.bytes_sz());
      vkUnmapMemory(device, staging.mem);

      buffer_copy(vi.bo.buf, staging.buf, nda->dims.bytes_sz());
      // XXX  wait for fence instead of idle -- var_to_nda as well
      vkFreeMemory(device, staging.mem, 0);
      vkDestroyBuffer(device, staging.buf, 0);

    }

    void copy_var_to_nda (p_nda_t const & nda, string const &vn) {

      vk_var_info_t const & vi = must_find( *vis, vn );
      assert( nda->dims == vi.dims);

      vk_buffer_info_t staging = create_buffer_info(nda->dims.bytes_sz(),
						      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
						      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      buffer_copy(staging.buf, vi.bo.buf, nda->dims.bytes_sz());
    
      void *dev_ptr;
      BAIL_ON_BAD_RESULT(vkMapMemory(device, staging.mem, 0, nda->dims.bytes_sz(), 0, (void **)&dev_ptr));
      memcpy( nda->rp_elems(), dev_ptr, nda->dims.bytes_sz());
      vkUnmapMemory(device, staging.mem);

      vkFreeMemory(device, staging.mem, 0);
      vkDestroyBuffer(device, staging.buf, 0);

    }
  
    p_nda_t get_var_raw_native_pointer( string const & vn ) {
      // XXX necessary?
      rt_err( "vk_compute_t: get_var_raw_native_pointer(): not implemented");
    }

    void create_var_with_dims( string const & vn, dims_t const & dims ) {
      vk_var_info_t var;
      vk_buffer_info_t bo = create_buffer_info(dims.bytes_sz(),
						 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
						 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
						 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      must_insert( *vis, vn, vk_var_info_t{bo, dims});

      set_var_to_zero( vn );
  
    }
  
    void create_var_with_dims_as_reshaped_view_of_var( string const & vn, dims_t const & dims, string const & src_vn ) {
    
      vk_var_info_t const & src_vi = must_find( *vis, src_vn );
      rtc_reshape_check( dims, src_vi.dims );
      must_insert( *vis, vn, vk_var_info_t{ src_vi.bo, dims } );
    }

    void release_var( string const & vn ) {
      vk_var_info_t var = must_find( *vis, vn );
      vkFreeMemory(device, var.bo.mem, 0);
      vkDestroyBuffer(device, var.bo.buf, 0);
      must_erase( *vis, vn );
    }
    virtual void release_all_funcs( void ) {
      for (auto& f : *kerns) {
	vkDestroyShaderModule(device, f.second.kern, 0);
      }
      kerns->clear();
    }

  
    dims_t get_var_dims( string const & vn ) { return must_find( *vis, vn ).dims; }

    void set_var_to_zero(string const & vn ) {
      vk_var_info_t var = must_find( *vis, vn );
      void* mem = malloc(var.dims.bytes_sz());
      memset(mem, 0, var.dims.bytes_sz());
      p_nda_t nda = make_shared<nda_t>(var.dims, mem);
      copy_nda_to_var(vn, nda);
      free(mem);
    }

  
    vk_compute_t( void ) : vis( new map_str_vk_var_info_t ), kerns( new map_str_vk_func_info_t ) { }

    vect_vk_query_pool_t call_evs;
    uint32_t alloc_call_id( void ) {
      // XXX actually use this, when running kernels
      VkQueryPoolCreateInfo query_info = {
	VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
	0,
	0,
	VK_QUERY_TYPE_TIMESTAMP,
	2,
	0
      };
      VkQueryPool query_pool;
      call_evs.push_back(query_pool);
    
      vkCreateQueryPool(device, &query_info, 0, &call_evs[call_evs.size()-1]);

      return call_evs.size() -1;
    }
    virtual void release_per_call_id_data( void ) {
      for (VkQueryPool pool : call_evs) {
	vkDestroyQueryPool(device, pool, 0);
      }
      call_evs.clear();
    } // invalidates all call_ids inside rtc_func_call_t's

    virtual float get_dur( uint32_t const & b, uint32_t const & e ) {
      // XXX implement
      return 0;
    }

    virtual float get_var_compute_dur( string const & vn ) { return 0; }
    virtual float get_var_ready_delta( string const & vn1, string const & vn2 ) { return 0; }

    void release_func( string const & func_name ) {

      vk_func_info_t func = must_find( *kerns, func_name );
      vkDestroyShaderModule(device, func.kern, 0);
      must_erase( *kerns, func_name );
    }

    uint32_t run( rtc_func_call_t const & rfc ) {
      // XXX remove debug timers
      timer_t t1("vk run");
      vk_func_info_t const & vfi = must_find(*kerns, rfc.rtc_func_name.c_str());
      uint32_t var_ix = 0;

      VkDescriptorSetLayoutBinding UBO_binding = {
	0,
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	1,
	VK_SHADER_STAGE_COMPUTE_BIT,
	0
      }; 
      vector<VkDescriptorSetLayoutBinding> var_bindings;
      vector<rtc_arg_t> UBO_args;
      vector<rtc_arg_t> var_args;
      size_t UBO_sz = 0;

    
      for( vect_string::const_iterator i = vfi.info.arg_names.begin(); i != vfi.info.arg_names.end(); ++i ) {  
	map_str_rtc_arg_t::const_iterator ai = rfc.arg_map.find( *i );
	if( ai == rfc.arg_map.end() )
	  { rt_err( strprintf( "vk_compute_t: arg '%s' not found in arg_map for call.\n",
			       str((*i)).c_str() ) ); }

	rtc_arg_t arg = ai->second;
	if (arg.is_var()) {
	  var_bindings.push_back({var_ix,
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		1,
		VK_SHADER_STAGE_COMPUTE_BIT,
		0});
	  var_args.push_back(arg);
	  var_ix++;
	  
	} else if (arg.is_nda()) {
	  UBO_args.push_back(arg);
	  // GLSL has no datatypes smaller than 4 bytes
	  UBO_sz += (arg.v->rp_elems() ? arg.v->dims.bytes_sz() : 4);
	}
      }

      VkDescriptorSetLayoutCreateInfo var_layout_create_info = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	0,
	0,
	var_ix,
	&var_bindings[0]
      };

      VkDescriptorSetLayout var_layout;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &var_layout_create_info, 0, &var_layout));

      VkDescriptorSetLayoutCreateInfo UBO_layout_create_info = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	0,
	0,
	1,
	&UBO_binding
      };

      VkDescriptorSetLayout UBO_layout;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &UBO_layout_create_info, 0, &UBO_layout));

      std::vector<VkDescriptorSetLayout> layouts;
      layouts.push_back(var_layout);
      layouts.push_back(UBO_layout);
    
      VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
	VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	0,
	0,
	2,
	&(layouts[0]),
	0,
	0
      };

      VkPipelineLayout pipeline_layout;
      BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &pipeline_layout_create_info, 0, &pipeline_layout));

      
      size_t const glob_work_sz = rfc.tpb.v*rfc.blks.v;
      size_t const loc_work_sz = rfc.tpb.v;

      
      const VkSpecializationMapEntry entries[] =
      // id,  offset,                size
        {{0, 0, sizeof(size_t)}};



  
      const VkSpecializationInfo spec_info = {
        1,                  // mapEntryCount
        entries,            // pMapEntries
        1 * sizeof(size_t),  // dataSize
        &loc_work_sz               // pData
      };

    
    
      VkComputePipelineCreateInfo compute_pipeline_create_info = {
	VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
	0,
	0,
	{
	  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	  0,
	  0,
	  VK_SHADER_STAGE_COMPUTE_BIT,
	  vfi.kern,
	  "main",
	  &spec_info
	},
	pipeline_layout,
	0,
	0
      };

      VkPipeline pipeline;
      BAIL_ON_BAD_RESULT(vkCreateComputePipelines(device, 0, 1, &compute_pipeline_create_info, 0, &pipeline));

  
      VkDescriptorPoolSize var_descriptor_pool_size = {
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	var_ix
      };


      VkDescriptorPoolCreateInfo var_descriptor_pool_create_info = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	0,
	0,
	1,
	1,
	&var_descriptor_pool_size
      };

      VkDescriptorPool var_descriptor_pool;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &var_descriptor_pool_create_info, 0, &var_descriptor_pool));

    
      VkDescriptorSetAllocateInfo var_descriptor_set_allocate_info = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	0,
	var_descriptor_pool,
	1,
	&var_layout
      };

      VkDescriptorSet var_descriptor_set;
      BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &var_descriptor_set_allocate_info, &var_descriptor_set));

      VkDescriptorPoolSize UBO_descriptor_pool_size = {
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	1
      };


      VkDescriptorPoolCreateInfo UBO_descriptor_pool_create_info = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	0,
	0,
	1,
	1,
	&UBO_descriptor_pool_size
      };

      VkDescriptorPool UBO_descriptor_pool;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &UBO_descriptor_pool_create_info, 0, &UBO_descriptor_pool));


      VkDescriptorSetAllocateInfo UBO_descriptor_set_allocate_info = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	0,
	UBO_descriptor_pool,
	1,
	&UBO_layout
      };

      VkDescriptorSet UBO_descriptor_set;
      BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &UBO_descriptor_set_allocate_info, &UBO_descriptor_set));

      vector<VkWriteDescriptorSet> descriptor_sets(var_ix+1);
      vector<VkDescriptorBufferInfo> bi(var_ix+1);
      var_ix = 0;

      for (rtc_arg_t arg : var_args) {
	VkBuffer buf = must_find(*vis, arg.n).bo.buf;
	bi[var_ix] = {
	  buf,
	  0,
	  VK_WHOLE_SIZE};

	descriptor_sets[var_ix] = {
	  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	  0,
	  var_descriptor_set,
	  var_ix,
	  0,
	  1,
	  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	  0,
	  &bi[var_ix],
	  0};
	var_ix++;
      }
      assert(var_ix == bi.size() -1);

      // XXX handle UBO_sz == 0
      vk_buffer_info_t ubo = create_buffer_info(UBO_sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
						  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      char* dev_ptr;
      BAIL_ON_BAD_RESULT(vkMapMemory(device, ubo.mem, 0, UBO_sz, 0, (void **) &dev_ptr));
      size_t offset = 0;
      for (rtc_arg_t arg : UBO_args) {
	if (arg.v->rp_elems()) {
	  memcpy(dev_ptr+offset, arg.v->rp_elems(), arg.v->dims.bytes_sz());
	  offset += arg.v->dims.bytes_sz();
	} else {
	  offset += 4;
	}
	
      }
      
      vkUnmapMemory(device, ubo.mem);
      
      bi[var_ix] = {
	ubo.buf,
	0,
	VK_WHOLE_SIZE
      };

      descriptor_sets[var_ix] = {
	VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	0,
	UBO_descriptor_set,
	0,
	0,
	1,
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	0,
	&bi[var_ix],
	0};
    
      vkUpdateDescriptorSets(device, var_ix+1, descriptor_sets.data(), 0, 0);

      uint32_t const call_id = alloc_call_id();

    
      VkCommandBufferAllocateInfo command_buffer_allocate_info = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	0,
	command_pool,
	VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	1
      };
      // move command_buffer to the class to avoid allocating a new CommandVuffer for every run
      VkCommandBuffer command_buffer;
      BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &command_buffer));

    
      VkCommandBufferBeginInfo command_buffer_begin_info = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	0,
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	0
      };

      BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info));

      // XXX use timestamps with queryPool from call_evs here to measure execution time

      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    
      std::vector<VkDescriptorSet> ds;
      ds.push_back(var_descriptor_set);
      ds.push_back(UBO_descriptor_set);

      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
			      pipeline_layout, 0, 2, &ds[0], 0, 0);
      vkCmdDispatch(command_buffer, glob_work_sz/loc_work_sz, 1, 1);

    
      BAIL_ON_BAD_RESULT(vkEndCommandBuffer(command_buffer));

    
      VkQueue queue;
      vkGetDeviceQueue(device, queue_family_index, 0, &queue);

      VkSubmitInfo submit_info = {
	VK_STRUCTURE_TYPE_SUBMIT_INFO,
	0,
	0,
	0,
	0,
	1,
	&command_buffer,
	0,
	0
      };

      timer_t t("vk kernel");
      BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submit_info, 0));
      BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));
      t.stop();
      
      vkFreeMemory(device, ubo.mem, 0);
      vkDestroyBuffer(device, ubo.buf, 0);
      vkDestroyDescriptorPool(device, var_descriptor_pool, 0);
      vkDestroyDescriptorPool(device, UBO_descriptor_pool, 0);
      vkDestroyDescriptorSetLayout(device, var_layout, 0);
      vkDestroyDescriptorSetLayout(device, UBO_layout, 0);
      vkDestroyPipelineLayout(device, pipeline_layout, 0);
      vkDestroyPipeline(device, pipeline, 0);

      return call_id;
    }

    void finish_and_sync( void ) { BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue)); }

    ~vk_compute_t() {
#ifdef DEBUG
      auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
      BAIL_ON_BAD_RESULT(func == nullptr ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS);
      func(instance, debug_report_callback, NULL);
#endif
      release_all_funcs();
      vkDestroyCommandPool(device, command_pool, 0);
      vkDestroyDevice(device, 0);
      vkDestroyInstance(instance, 0);
    }

    // FIXME: TODO
    void profile_start( void ) { }
    void profile_stop( void ) { }
  
  };

#include"gen/vk_util.cc.nesi_gen.cc"
}
