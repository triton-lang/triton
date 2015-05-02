#*********************************************************#
#*  File: Apk.cmake                                      *
#*    Android apk tools
#*
#*  Copyright (C) 2002-2013 The PixelLight Team (http://www.pixellight.org/)
#*
#*  This file is part of PixelLight.
#*
#*  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#*  and associated documentation files (the "Software"), to deal in the Software without
#*  restriction, including without limitation the rights to use, copy, modify, merge, publish,
#*  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
#*  Software is furnished to do so, subject to the following conditions:
#*
#*  The above copyright notice and this permission notice shall be included in all copies or
#*  substantial portions of the Software.
#*
#*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#*  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#*  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#*  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#*  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#*********************************************************#


##################################################
## Options
##################################################
set(ANDROID_APK_CREATE				"1"							CACHE BOOL		"Create apk file?")
set(ANDROID_APK_INSTALL				"1"							CACHE BOOL		"Install created apk file on the device automatically?")
set(ANDROID_APK_RUN					"1"							CACHE BOOL		"Run created apk file on the device automatically? (installs it automatically as well, \"ANDROID_APK_INSTALL\"-option is ignored)")
set(ANDROID_APK_TOP_LEVEL_DOMAIN	"org"						CACHE STRING	"Top level domain name of the organization (follow the package naming conventions (http://en.wikipedia.org/wiki/Java_package#Package_naming_conventions))")
set(ANDROID_APK_DOMAIN				"pixellight"				CACHE STRING	"Organization's domain (follow the package naming conventions (http://en.wikipedia.org/wiki/Java_package#Package_naming_conventions))")
set(ANDROID_APK_SUBDOMAIN			"test"						CACHE STRING	"Any subdomains (follow the package naming conventions (http://en.wikipedia.org/wiki/Java_package#Package_naming_conventions))")
set(ANDROID_APK_FULLSCREEN			"1"							CACHE BOOL		"Run the application in fullscreen? (no status/title bar)")
set(ANDROID_APK_RELEASE				"0"							CACHE BOOL		"Create apk file ready for release? (signed, you have to enter a password during build, do also setup \"ANDROID_APK_SIGNER_KEYSTORE\" and \"ANDROID_APK_SIGNER_ALIAS\")")
set(ANDROID_APK_SIGNER_KEYSTORE		"~/my-release-key.keystore"	CACHE STRING	"Keystore for signing the apk file (only required for release apk)")
set(ANDROID_APK_SIGNER_ALIAS		"myalias"					CACHE STRING	"Alias for signing the apk file (only required for release apk)")


##################################################
## Variables
##################################################
set(ANDROID_THIS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})	# Directory this CMake file is in


##################################################
## MACRO: android_create_apk
##
## Create/copy Android apk related files
##
## @param name
##   Name of the project (e.g. "MyProject"), this will also be the name of the created apk file
## @param apk_directory
##   Directory were to construct the apk file in (e.g. "${CMAKE_BINARY_DIR}/apk")
## @param shared_libraries
##   List of shared libraries (absolute filenames) this application is using, these libraries are copied into the apk file and will be loaded automatically within a generated Java file - Lookout! The order is important due to shared library dependencies!
## @param assets
##   List of assets to copy into the apk file (absolute filenames, wildcards like "*.*" are allowed)
## @param data_directory
##   Subdirectory within the apk asset directory to copy the "assets"-files into (e.g. "Data")
##
## @remarks
##   Requires the following tools to be found automatically
##   - "android" (part of the Android SDK)
##   - "adb" (part of the Android SDK)
##   - "ant" (type e.g. "sudo apt-get install ant" on your Linux system to install Ant)
##   - "jarsigner" (part of the JDK)
##   - "zipalign" (part of the Android SDK)
##################################################
macro(android_create_apk name apk_directory shared_libraries assets data_directory)
	if(ANDROID_APK_CREATE)
		# Construct the current package name and theme
		set(ANDROID_APK_PACKAGE "${ANDROID_APK_TOP_LEVEL_DOMAIN}.${ANDROID_APK_DOMAIN}.${ANDROID_APK_SUBDOMAIN}")
		if(ANDROID_APK_FULLSCREEN)
			set(ANDROID_APK_THEME "android:theme=\"@android:style/Theme.NoTitleBar.Fullscreen\"")
		else()
			set(ANDROID_APK_THEME "")
		endif()
		set(ANDROID_NAME ${name})
		if(CMAKE_BUILD_TYPE MATCHES Debug)
			set(ANDROID_APK_DEBUGGABLE "true")
			set(ANDROID_APK_RELEASE_LOCAL "0")
		else()
			set(ANDROID_APK_DEBUGGABLE "false")
			set(ANDROID_APK_RELEASE_LOCAL ${ANDROID_APK_RELEASE})
		endif()

		# Create "AndroidManifest.xml"
		configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android/AndroidManifest.xml.in" "${apk_directory}/AndroidManifest.xml")

		# Create "res/values/strings.xml"
		configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android/strings.xml.in" "${apk_directory}/res/values/strings.xml")

		# Get a list of libraries to load in (e.g. "PLCore;PLMath" etc.)
		set(ANDROID_SHARED_LIBRARIES_TO_LOAD "")
		foreach(value ${shared_libraries})
			# "value" is e.g. "/home/cofenberg/pl_ndk/Bin-Linux-ndk/Runtime/armeabi/libPLCore.so"
			get_filename_component(shared_library_filename ${value} NAME_WE)

			# "shared_library_filename" is e.g. "libPLCore", but we need "PLCore"
			string(LENGTH ${shared_library_filename} shared_library_filename_length)
			math(EXPR shared_library_filename_length ${shared_library_filename_length}-3)
			string(SUBSTRING ${shared_library_filename} 3 ${shared_library_filename_length} shared_library_filename)

			# "shared_library_filename" is now e.g. "PLCore", this is what we want -> Add it to the list
			set(ANDROID_SHARED_LIBRARIES_TO_LOAD ${ANDROID_SHARED_LIBRARIES_TO_LOAD} ${shared_library_filename})
		endforeach()

		# Create Java file which is responsible for loading in the required shared libraries (the content of "ANDROID_SHARED_LIBRARIES_TO_LOAD" is used for this)
		configure_file("${CMAKE_CURRENT_SOURCE_DIR}/android/LoadLibraries.java.in" "${apk_directory}/src/${ANDROID_APK_TOP_LEVEL_DOMAIN}/${ANDROID_APK_DOMAIN}/${ANDROID_APK_SUBDOMAIN}/LoadLibraries.java")

		# Create the directory for the libraries
 		add_custom_command(TARGET ${ANDROID_NAME}
			PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E remove_directory "${apk_directory}/libs"
		)
		add_custom_command(TARGET ${ANDROID_NAME}
			PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory "${apk_directory}/libs/${ARM_TARGET}"
		)

		# Copy the used shared libraries
		foreach(value ${shared_libraries})
			add_custom_command(TARGET ${ANDROID_NAME}
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy ${value} "${apk_directory}/libs/${ARM_TARGET}"
			)
		endforeach()

		# Create "build.xml", "default.properties", "local.properties" and "proguard.cfg" files
		add_custom_command(TARGET ${ANDROID_NAME}
			COMMAND android update project -t android-${ANDROID_API_LEVEL} --name ${ANDROID_NAME} --path "${apk_directory}"
		)

		# Copy assets
		add_custom_command(TARGET ${ANDROID_NAME}
			PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E remove_directory "${apk_directory}/assets"
		)
		add_custom_command(TARGET ${ANDROID_NAME}
			PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory "${apk_directory}/assets/${data_directory}"
		)
		#foreach(value ${assets})
		#	android_copy_files(${value} "${apk_directory}/assets/${data_directory}")
		#endforeach()

		# In case of debug build, do also copy gdbserver
		#if(CMAKE_BUILD_TYPE MATCHES Debug)
		#	add_custom_command(TARGET ${ANDROID_NAME}
		#		POST_BUILD
		#		COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_GDBSERVER} "${apk_directory}/libs/${ARM_TARGET}"
		#	)
		#endif()

		# Uninstall previous version from the device/emulator (else we may get e.g. signature conflicts)
		add_custom_command(TARGET ${ANDROID_NAME}
			COMMAND adb uninstall ${ANDROID_APK_PACKAGE}
		)

		# Build the apk file
		if(ANDROID_APK_RELEASE_LOCAL)
			# Let Ant create the unsigned apk file
			add_custom_command(TARGET ${ANDROID_NAME}
				COMMAND ant release
				WORKING_DIRECTORY "${apk_directory}"
			)

			# Sign the apk file
			add_custom_command(TARGET ${ANDROID_NAME}
				COMMAND jarsigner -verbose -keystore ${ANDROID_APK_SIGNER_KEYSTORE} bin/${ANDROID_NAME}-unsigned.apk ${ANDROID_APK_SIGNER_ALIAS}
				WORKING_DIRECTORY "${apk_directory}"
			)

			# Align the apk file
			add_custom_command(TARGET ${ANDROID_NAME}
				COMMAND zipalign -v -f 4 bin/${ANDROID_NAME}-unsigned.apk bin/${ANDROID_NAME}.apk
				WORKING_DIRECTORY "${apk_directory}"
			)

			# Install current version on the device/emulator
			if(ANDROID_APK_INSTALL OR ANDROID_APK_RUN)
				add_custom_command(TARGET ${ANDROID_NAME}
					COMMAND adb install -r bin/${ANDROID_NAME}.apk
					WORKING_DIRECTORY "${apk_directory}"
				)
			endif()
		else()
			# Let Ant create the unsigned apk file
			add_custom_command(TARGET ${ANDROID_NAME}
				COMMAND ant debug
				WORKING_DIRECTORY "${apk_directory}"
			)

			# Install current version on the device/emulator
			if(ANDROID_APK_INSTALL OR ANDROID_APK_RUN)
				add_custom_command(TARGET ${ANDROID_NAME}
					COMMAND adb install -r bin/${ANDROID_NAME}-debug.apk
					WORKING_DIRECTORY "${apk_directory}"
				)
			endif()
		endif()

		# Start the application
		if(ANDROID_APK_RUN)
			add_custom_command(TARGET ${ANDROID_NAME}
				COMMAND adb shell am start -n ${ANDROID_APK_PACKAGE}/${ANDROID_APK_PACKAGE}.LoadLibraries
			)
		endif()
	endif()
endmacro(android_create_apk name apk_directory shared_libraries assets data_directory)

##################################################
## MACRO: android_copy_files
##
## Copy files from one place to another using wildcards
##################################################
macro(android_copy_files src dest)
	# Get path
	get_filename_component(path ${src} PATH)

	# Get exclude option
	set(exclude)
	if("${ARGV2}" STREQUAL "EXCLUDE")
		set(exclude ${ARGV3})
	endif()

	# Find files
	file(GLOB_RECURSE files RELATIVE ${path} ${src})

	# Find files
	foreach(file ${files})
		# Get source and destination file
		set(src_file ${path}/${file})
		set(dst_file ${dest}/${file})

		# Check exclude expression
		set(copy 1)
		if(exclude)
			if(file MATCHES ${exclude})
				set(copy 0)
			endif()
		endif()

		# Process file
		if(copy EQUAL 1)
			# Create output directory
			get_filename_component(dst_path ${dst_file} PATH)
			file(MAKE_DIRECTORY ${dst_path})

			# Copy file
			add_custom_command(TARGET ${ANDROID_NAME}
				PRE_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy ${src_file} ${dst_file} VERBATIM
			)
		endif()
	endforeach(file ${files})
endmacro(android_copy_files src dest)
