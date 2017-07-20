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
  struct vk_buffer_object_t {
    VkBuffer buf;
    VkDeviceMemory mem;
  };

  typedef vector<vk_buffer_object_t> vect_vk_buffer_object_t;

  struct vk_var_info_t {
    vk_buffer_object_t bo;
    dims_t dims;
  };

  typedef map < string, vk_var_info_t > map_str_vk_var_info_t;
  typedef shared_ptr< map_str_vk_var_info_t > p_map_str_vk_var_info_t;

  typedef vector< VkQueryPool > vect_vk_query_pool_t;

  struct vk_compute_t : virtual public nesi, public rtc_compute_t // NESI(help="Vulkan based rtc support",
			// bases=["rtc_compute_t"], type_id="vk" )
  {
  
    VkInstance instance;
    
    VkDebugReportCallbackEXT debugReportCallback;
    
    VkDevice device;

    VkPhysicalDeviceMemoryProperties memProperties;

    VkQueue queue;

    uint32_t queueFamilyIndex;

    VkCommandPool commandPool;
    zi_bool init_done;

    p_map_str_vk_var_info_t vis;
    p_map_str_vk_func_info_t kerns;

    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support

  
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
							VkDebugReportFlagsEXT flags,
							VkDebugReportObjectTypeEXT objType,
							uint64_t obj,
							size_t location,
							int32_t code,
							const char* layerPrefix,
							const char* msg,
							void* userData) {

      std::cerr << "validation layer: " << msg << std::endl;

      return VK_FALSE;
    }
    
    void init( void ) {
    
      const VkApplicationInfo applicationInfo = {
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
      const VkInstanceCreateInfo instanceCreateInfo = {
	VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	0,
	0,
	&applicationInfo,
	layers.size(),
	layers.data(),
	extensions.size(),
	extensions.data(),
      };

      BAIL_ON_BAD_RESULT(vkCreateInstance(&instanceCreateInfo, 0, &instance));

#ifdef DEBUG
      const VkDebugReportCallbackCreateInfoEXT debugCallbackCreateInfo = {
	VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
	0,
	VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
	debugCallback,
	0};

      // We have to explicitly load this function.
      auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
      BAIL_ON_BAD_RESULT(vkCreateDebugReportCallbackEXT == nullptr ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS);

      // Create and register callback.
      BAIL_ON_BAD_RESULT(vkCreateDebugReportCallbackEXT(instance, &debugCallbackCreateInfo, NULL, &debugReportCallback));
#endif
    
      uint32_t physicalDeviceCount = 0;
      BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0));

      VkPhysicalDevice* const physicalDevices = (VkPhysicalDevice*)malloc(
									  sizeof(VkPhysicalDevice) * physicalDeviceCount);

      BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices));

      BAIL_ON_BAD_RESULT(physicalDeviceCount ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED);

      uint32_t queueFamilyCount;

      // XXX just use device 0 for now

      vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueFamilyCount, 0);
      std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);

      vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueFamilyCount, queueFamilies.data());
      queueFamilyIndex = 0;
      bool foundQueueFamily = false;
      for (; queueFamilyIndex < queueFamilyCount; queueFamilyIndex++) {
	VkQueueFamilyProperties props = queueFamilies[queueFamilyIndex];
      
	if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
	  foundQueueFamily = true;
	  break;
	}
      }

      BAIL_ON_BAD_RESULT(foundQueueFamily ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED);

      // XXX investigate
      const float queuePriority = 1.0f;

    
      const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
	VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	0,
	0,
	queueFamilyIndex,
	1,
	&queuePriority
      };

      // XXX device extensions and layers
      const VkDeviceCreateInfo deviceCreateInfo = {
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

      BAIL_ON_BAD_RESULT(vkCreateDevice(physicalDevices[0], &deviceCreateInfo, 0, &device));

      vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &memProperties);
  
      vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    

      VkCommandPoolCreateInfo commandPoolCreateInfo = {
	VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	0,
	0,
	queueFamilyIndex
      };

      BAIL_ON_BAD_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool));

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
	  p_ostream out = ofs_open( strprintf( "%s/%s_%d.glsl", gen_src_output_dir.exp.c_str(), i->func_name.c_str(), i - func_infos.begin() ));
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
    
	VkShaderModuleCreateInfo shaderModuleCreateInfo = {
	  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	  0,
	  0,
	  buffer.size() * sizeof(uint32_t),
	  buffer.data()
	};

	VkShaderModule shaderModule;
	BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shaderModule));

	must_insert(*kerns, i->func_name, vk_func_info_t{*i, shaderModule});
      }      

    
    }

    void copy_nda_to_var (string const &vn, p_nda_t const & nda) {

      // XXX maybe we should use some kind of heuristic to avoid allocating
      // a new staging buffer everytime - var_to_nda as well
      vk_var_info_t const & vi = must_find( *vis, vn );
      assert( nda->dims == vi.dims);

      
      const VkBufferCreateInfo bufferCreateInfo = {
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	0,
	0,
	nda->dims.bytes_sz(),
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	VK_SHARING_MODE_EXCLUSIVE,
	1,
	&queueFamilyIndex
      };

    
      VkBuffer buf;
      BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &buf));

      
      // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
      uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

      VkMemoryRequirements memReqs;
      vkGetBufferMemoryRequirements(device, buf, &memReqs);

      // host visible memory

      for (uint32_t k = 0; k < memProperties.memoryTypeCount; k++) {
	if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memProperties.memoryTypes[k].propertyFlags) &&
	    (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memProperties.memoryTypes[k].propertyFlags) &&
	    (memReqs.memoryTypeBits & (1 << k))) {
	  memoryTypeIndex = k;
	  break;
	}
      }

      BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

      const VkMemoryAllocateInfo memoryAllocateInfo = {
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	0,
	nda->dims.bytes_sz(),
	memoryTypeIndex
      };
      VkDeviceMemory devMem;
      BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, 0, &devMem));

      void *devPtr;
      BAIL_ON_BAD_RESULT(vkMapMemory(device, devMem, 0, nda->dims.bytes_sz(), 0, (void **)&devPtr));
      memcpy(devPtr, nda->rp_elems(), nda->dims.bytes_sz());
      vkUnmapMemory(device, devMem);


      BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, buf, devMem, 0));

      VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	0,
	commandPool,
	VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	1
      };

      VkCommandBuffer commandBuffer;
      BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));

      VkCommandBufferBeginInfo commandBufferBeginInfo = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	0,
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	0
      };

      BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

      VkBufferCopy bufferCopy = {0, 0, nda->dims.bytes_sz()};
      vkCmdCopyBuffer(commandBuffer, buf, vi.bo.buf, 1, &bufferCopy);


      BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));


      VkSubmitInfo submitInfo = {
	VK_STRUCTURE_TYPE_SUBMIT_INFO,
	0,
	0,
	0,
	0,
	1,
	&commandBuffer,
	0,
	0
      };
      BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));
      BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));

      // XXX  wait for fence instead of idle -- var_to_nda as well
      vkFreeMemory(device, devMem, 0);
      vkDestroyBuffer(device, buf, 0);

    }

    void copy_var_to_nda (p_nda_t const & nda, string const &vn) {

      vk_var_info_t const & vi = must_find( *vis, vn );
      assert( nda->dims == vi.dims);

      const VkBufferCreateInfo bufferCreateInfo = {
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	0,
	0,
	nda->dims.bytes_sz(),
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	VK_SHARING_MODE_EXCLUSIVE,
	1,
	&queueFamilyIndex
      };
  
      VkBuffer buf;
      BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &buf));
      
      // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
      uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

      VkMemoryRequirements memReqs;
      vkGetBufferMemoryRequirements(device, buf, &memReqs);

      // host visible memory

      for (uint32_t k = 0; k < memProperties.memoryTypeCount; k++) {
	if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memProperties.memoryTypes[k].propertyFlags) &&
	    (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memProperties.memoryTypes[k].propertyFlags) &&
	    (memReqs.memoryTypeBits & (1 << k))) {
	  memoryTypeIndex = k;
	  break;
	}
      }

      BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

      const VkMemoryAllocateInfo memoryAllocateInfo = {
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	0,
        memReqs.size,
	memoryTypeIndex
      };
      VkDeviceMemory devMem;
      BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, 0, &devMem));

      BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, buf, devMem, 0));

      VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	0,
	commandPool,
	VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	1
      };

      VkCommandBuffer commandBuffer;
      BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));

      VkCommandBufferBeginInfo commandBufferBeginInfo = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	0,
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	0
      };

      BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

      VkBufferCopy bufferCopy = {0, 0, nda->dims.bytes_sz()};
      vkCmdCopyBuffer(commandBuffer, vi.bo.buf, buf, 1, &bufferCopy);


      BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));


      VkSubmitInfo submitInfo = {
	VK_STRUCTURE_TYPE_SUBMIT_INFO,
	0,
	0,
	0,
	0,
	1,
	&commandBuffer,
	0,
	0
      };
      BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));

      // XXX use fence
      BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));

    
      void *devPtr;
      BAIL_ON_BAD_RESULT(vkMapMemory(device, devMem, 0, nda->dims.bytes_sz(), 0, (void **)&devPtr));
      memcpy( nda->rp_elems(), devPtr, nda->dims.bytes_sz());
      vkUnmapMemory(device, devMem);
      float* res = (float*)nda->rp_elems();

      vkFreeMemory(device, devMem, 0);
      vkDestroyBuffer(device, buf, 0);

    }
  
    p_nda_t get_var_raw_native_pointer( string const & vn ) {
      // XXX necessary?
      rt_err( "vk_compute_t: get_var_raw_native_pointer(): not implemented");
    }

    void create_var_with_dims( string const & vn, dims_t const & dims ) {
      vk_var_info_t var;
      
      const VkBufferCreateInfo bufferCreateInfo = {
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	0,
	0,
	dims.bytes_sz(),
	// XXX investigate src dst specialization
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	VK_SHARING_MODE_EXCLUSIVE,
	1,
	&queueFamilyIndex
      };

    
      VkBuffer buf;
      BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &buf));

      // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
      uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;


      VkMemoryRequirements memReqs;
      vkGetBufferMemoryRequirements(device, buf, &memReqs);

      
      // device local memory

      for (uint32_t k = 0; k < memProperties.memoryTypeCount; k++) {
	if ((VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT & memProperties.memoryTypes[k].propertyFlags) &&
	    (memReqs.memoryTypeBits & (1 << k))) {
	  memoryTypeIndex = k;
	  break;
	}
      }

      BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

      const VkMemoryAllocateInfo memoryAllocateInfo = {
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	0,
	memReqs.size,
	memoryTypeIndex
      };
      VkDeviceMemory devMem;
      BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, 0, &devMem));

      BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, buf, devMem, 0));

      must_insert( *vis, vn, vk_var_info_t{vk_buffer_object_t{buf, devMem}, dims});

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
      VkQueryPoolCreateInfo queryInfo = {
	VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
	0,
	0,
	VK_QUERY_TYPE_TIMESTAMP,
	2,
	0
      };
      VkQueryPool queryPool;
      call_evs.push_back(queryPool);
    
      vkCreateQueryPool(device, &queryInfo, 0, &call_evs[call_evs.size()-1]);

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
      int varIx = 0;

      VkDescriptorSetLayoutBinding UBOBinding = {
	0,
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	1,
	VK_SHADER_STAGE_COMPUTE_BIT,
	0
      }; 
      vector<VkDescriptorSetLayoutBinding> varBindings;
      vector<rtc_arg_t> UBOArgs;
      vector<rtc_arg_t> varArgs;
      size_t UBO_sz = 0;

    
      for( vect_string::const_iterator i = vfi.info.arg_names.begin(); i != vfi.info.arg_names.end(); ++i ) {  
	map_str_rtc_arg_t::const_iterator ai = rfc.arg_map.find( *i );
	if( ai == rfc.arg_map.end() )
	  { rt_err( strprintf( "vk_compute_t: arg '%s' not found in arg_map for call.\n",
			       str((*i)).c_str() ) ); }

	rtc_arg_t arg = ai->second;
	if (arg.is_var()) {
	  varBindings.push_back({varIx,
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		1,
		VK_SHADER_STAGE_COMPUTE_BIT,
		0});
	  varArgs.push_back(arg);
	  varIx++;
	  
	} else if (arg.is_nda()) {
	  UBOArgs.push_back(arg);
	  // GLSL has no datatypes smaller than 4 bytes
	  UBO_sz += (arg.v->rp_elems() ? arg.v->dims.bytes_sz() : 4);
	}
      }

      VkDescriptorSetLayoutCreateInfo varLayoutCreateInfo = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	0,
	0,
	varIx,
	&varBindings[0]
      };

      VkDescriptorSetLayout varLayout;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &varLayoutCreateInfo, 0, &varLayout));

      VkDescriptorSetLayoutCreateInfo UBOLayoutCreateInfo = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	0,
	0,
	1,
	&UBOBinding
      };

      VkDescriptorSetLayout UBOLayout;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &UBOLayoutCreateInfo, 0, &UBOLayout));

      std::vector<VkDescriptorSetLayout> layouts;
      layouts.push_back(varLayout);
      layouts.push_back(UBOLayout);
    
      VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
	VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	0,
	0,
	2,
	&(layouts[0]),
	0,
	0
      };

      VkPipelineLayout pipelineLayout;
      BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, 0, &pipelineLayout));

      
      size_t const glob_work_sz = rfc.tpb.v*rfc.blks.v;
      size_t const loc_work_sz = rfc.tpb.v;

      
      const VkSpecializationMapEntry entries[] =
      // id,  offset,                size
        {{0, 0, sizeof(size_t)}};



  
      const VkSpecializationInfo specInfo = {
        1,                  // mapEntryCount
        entries,            // pMapEntries
        1 * sizeof(size_t),  // dataSize
        &loc_work_sz               // pData
      };

    
    
      VkComputePipelineCreateInfo computePipelineCreateInfo = {
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
	  &specInfo
	},
	pipelineLayout,
	0,
	0
      };

      VkPipeline pipeline;
      BAIL_ON_BAD_RESULT(vkCreateComputePipelines(device, 0, 1, &computePipelineCreateInfo, 0, &pipeline));

  
      VkDescriptorPoolSize varDescriptorPoolSize = {
	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	varIx
      };


      VkDescriptorPoolCreateInfo varDescriptorPoolCreateInfo = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	0,
	0,
	1,
	1,
	&varDescriptorPoolSize
      };

      VkDescriptorPool varDescriptorPool;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &varDescriptorPoolCreateInfo, 0, &varDescriptorPool));

    
      VkDescriptorSetAllocateInfo varDescriptorSetAllocateInfo = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	0,
	varDescriptorPool,
	1,
	&varLayout
      };

      VkDescriptorSet varDescriptorSet;
      BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &varDescriptorSetAllocateInfo, &varDescriptorSet));

      VkDescriptorPoolSize UBODescriptorPoolSize = {
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	1
      };


      VkDescriptorPoolCreateInfo UBODescriptorPoolCreateInfo = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	0,
	0,
	1,
	1,
	&UBODescriptorPoolSize
      };

      VkDescriptorPool UBODescriptorPool;
      BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &UBODescriptorPoolCreateInfo, 0, &UBODescriptorPool));


      VkDescriptorSetAllocateInfo UBODescriptorSetAllocateInfo = {
	VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	0,
	UBODescriptorPool,
	1,
	&UBOLayout
      };

      VkDescriptorSet UBODescriptorSet;
      BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &UBODescriptorSetAllocateInfo, &UBODescriptorSet));

      vector<VkWriteDescriptorSet> descriptorSets(varIx+1);
      vector<VkDescriptorBufferInfo> bi(varIx+1);
      varIx = 0;

      for (rtc_arg_t arg : varArgs) {
	VkBuffer buf = must_find(*vis, arg.n).bo.buf;
	bi[varIx] = {
	  buf,
	  0,
	  VK_WHOLE_SIZE};

	descriptorSets[varIx] = {
	  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	  0,
	  varDescriptorSet,
	  varIx,
	  0,
	  1,
	  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	  0,
	  &bi[varIx],
	  0};
	varIx++;
      }
      assert(varIx == bi.size() -1);
      

      //create buffer to store ndas into
      const VkBufferCreateInfo uboCreateInfo = {
	VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	0,
	0,
	UBO_sz, // XXX handle UBO_sz == 0
	VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	VK_SHARING_MODE_EXCLUSIVE,
	1,
	&queueFamilyIndex
      };

      VkBuffer UBOBuffer;
      BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &uboCreateInfo, 0, &UBOBuffer));

      
      VkMemoryRequirements memReqs;
      vkGetBufferMemoryRequirements(device, UBOBuffer, &memReqs);

      uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;
      // XXX we can use a staging buffer here as well, if it affects perf
      for (uint32_t k = 0; k < memProperties.memoryTypeCount; k++) {
	if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memProperties.memoryTypes[k].propertyFlags) &&
	    (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memProperties.memoryTypes[k].propertyFlags) &&
	    UBO_sz < memProperties.memoryHeaps[memProperties.memoryTypes[k].heapIndex].size &&
	    (memReqs.memoryTypeBits & (1 << k))) {
	  memoryTypeIndex = k;
	  break;
	}
      }
      BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY
			 : VK_SUCCESS);
      VkMemoryAllocateInfo memoryAllocateInfo = {
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	0,
	memReqs.size,
	memoryTypeIndex
      };
      VkDeviceMemory UBOMemory;
      BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, 0, &UBOMemory));

      char* devPtr;
      BAIL_ON_BAD_RESULT(vkMapMemory(device, UBOMemory, 0, UBO_sz, 0, (void **) &devPtr));
      size_t offset = 0;
      for (rtc_arg_t arg : UBOArgs) {
	if (arg.v->rp_elems()) {
	  memcpy(devPtr+offset, arg.v->rp_elems(), arg.v->dims.bytes_sz());
	  offset += arg.v->dims.bytes_sz();
	} else {
	  offset += 4;
	}
	
      }
      
      vkUnmapMemory(device, UBOMemory);
      
      BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, UBOBuffer, UBOMemory, 0));

      bi[varIx] = {
	UBOBuffer,
	0,
	VK_WHOLE_SIZE
      };

      descriptorSets[varIx] = {
	VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
	0,
	UBODescriptorSet,
	0,
	0,
	1,
	VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	0,
	&bi[varIx],
	0};
    
      vkUpdateDescriptorSets(device, varIx+1, descriptorSets.data(), 0, 0);

      uint32_t const call_id = alloc_call_id();

    
      VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	0,
	commandPool,
	VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	1
      };
      // move commandBuffer to the class to avoid allocating a new CommandVuffer for every run
      VkCommandBuffer commandBuffer;
      BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));

    
      VkCommandBufferBeginInfo commandBufferBeginInfo = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	0,
	VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	0
      };

      BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

      // XXX use timestamps with queryPool from call_evs here to measure execution time

      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    
      std::vector<VkDescriptorSet> ds;
      ds.push_back(varDescriptorSet);
      ds.push_back(UBODescriptorSet);

      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
			      pipelineLayout, 0, 2, &ds[0], 0, 0);
      vkCmdDispatch(commandBuffer, glob_work_sz/loc_work_sz, 1, 1);

    
      BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));

    
      VkQueue queue;
      vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

      VkSubmitInfo submitInfo = {
	VK_STRUCTURE_TYPE_SUBMIT_INFO,
	0,
	0,
	0,
	0,
	1,
	&commandBuffer,
	0,
	0
      };

      timer_t t("vk kernel");
      BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));
      BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));
      t.stop();
      
      vkFreeMemory(device, UBOMemory, 0);
      vkDestroyBuffer(device, UBOBuffer, 0);
      vkDestroyDescriptorPool(device, varDescriptorPool, 0);
      vkDestroyDescriptorPool(device, UBODescriptorPool, 0);
      vkDestroyDescriptorSetLayout(device, varLayout, 0);
      vkDestroyDescriptorSetLayout(device, UBOLayout, 0);
      vkDestroyPipelineLayout(device, pipelineLayout, 0);
      vkDestroyPipeline(device, pipeline, 0);

      return call_id;
    }

    void finish_and_sync( void ) { BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue)); }

    ~vk_compute_t() {
#ifdef DEBUG
      auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
      BAIL_ON_BAD_RESULT(func == nullptr ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS);
      func(instance, debugReportCallback, NULL);
#endif
      release_all_funcs();
      vkDestroyCommandPool(device, commandPool, 0);
      vkDestroyDevice(device, 0);
      vkDestroyInstance(instance, 0);
    }

    // FIXME: TODO
    void profile_start( void ) { }
    void profile_stop( void ) { }
  
  };

#include"gen/vk_util.cc.nesi_gen.cc"
}
