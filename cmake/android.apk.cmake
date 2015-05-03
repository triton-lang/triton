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
set(ANDROID_APK_API_LEVEL "10" CACHE STRING "Android APK API level")
set(ANDROID_APK_INSTALL "0" CACHE BOOL "Install created apk file on the device automatically?")
set(ANDROID_APK_RUN "0" CACHE BOOL "Run created apk file on the device automatically? (installs it automatically as well, \"ANDROID_APK_INSTALL\"-option is ignored)")
set(ANDROID_APK_SIGNER_KEYSTORE	"~/my-release-key.keystore" CACHE STRING "Keystore for signing the apk file (only required for release apk)")
set(ANDROID_APK_SIGNER_ALIAS "myalias" CACHE STRING "Alias for signing the apk file (only required for release apk)")

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
## @param apk_pacakge_name
##   Pacakge name of the application
## @param apk_directory
##   Directory were to construct the apk file in (e.g. "${CMAKE_BINARY_DIR}/apk")
## @param libs_directory
##   Directory where the built android libraries will be POST_BUILD, e.g ${CMAKE_SOURCE_DIR}/libs 
## @param assets_directory
##   Directory where the assets for the application are locatated
##   
## @remarks
##   Requires the following tools to be found automatically
##   - "android" (part of the Android SDK)
##   - "adb" (part of the Android SDK)
##   - "ant" (type e.g. "sudo apt-get install ant" on your Linux system to install Ant)
##   - "jarsigner" (part of the JDK)
##   - "zipalign" (part of the Android SDK)
##################################################


macro(android_create_apk name apk_package_name apk_directory libs_directory android_directory assets_directory)
  set(ANDROID_NAME ${name})
  set(ANDROID_APK_PACKAGE ${apk_package_name})

  # Create the directory for the libraries
  add_custom_command(TARGET ${ANDROID_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E remove_directory "${apk_directory}/libs")
  add_custom_command(TARGET ${ANDROID_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "${apk_directory}/libs/armeabi-v7a/")
  get_property(MAINLIB TARGET ${name} PROPERTY LOCATION)
  get_property(ISAAC TARGET isaac PROPERTY LOCATION)
  add_custom_command(TARGET ${ANDROID_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy  "${MAINLIB}" "${apk_directory}/libs/armeabi-v7a/")

  # Create "build.xml", "default.properties", "local.properties" and "proguard.cfg" files
  if(CMAKE_BUILD_TYPE MATCHES Release)
    set(ANDROID_APK_DEBUGGABLE "false")
  else()
    set(ANDROID_APK_DEBUGGABLE "true")
  endif()

  add_custom_command(TARGET ${ANDROID_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "${apk_directory}/res")
  add_custom_command(TARGET ${ANDROID_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory  "${android_directory}/res" "${apk_directory}/res/")

  configure_file("${android_directory}/AndroidManifest.xml" "${apk_directory}/AndroidManifest.xml")

  add_custom_command(TARGET ${ANDROID_NAME} COMMAND android update project -t android-${ANDROID_APK_API_LEVEL} --name ${ANDROID_NAME} --path "${apk_directory}")

  # Copy assets
  add_custom_command(TARGET ${ANDROID_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E remove_directory "${apk_directory}/assets")
  add_custom_command(TARGET ${ANDROID_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "${apk_directory}/assets/")
  add_custom_command(TARGET ${ANDROID_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory  "${CMAKE_SOURCE_DIR}/assets" "${apk_directory}/assets/")

  # Build the apk file
  if(CMAKE_BUILD_TYPE MATCHES Release)
    # Let Ant create the unsigned apk file
    add_custom_command(TARGET ${ANDROID_NAME}
      COMMAND ant release
      WORKING_DIRECTORY "${apk_directory}")

    # Sign the apk file
    add_custom_command(TARGET ${ANDROID_NAME}
      COMMAND jarsigner -verbose -keystore ${ANDROID_APK_SIGNER_KEYSTORE} bin/${ANDROID_NAME}-unsigned.apk ${ANDROID_APK_SIGNER_ALIAS}
      WORKING_DIRECTORY "${apk_directory}")

    # Align the apk file
    add_custom_command(TARGET ${ANDROID_NAME}
      COMMAND zipalign -v -f 4 bin/${ANDROID_NAME}-unsigned.apk bin/${ANDROID_NAME}.apk
      WORKING_DIRECTORY "${apk_directory}")
    
    # Install current version on the device/emulator
    if(ANDROID_APK_INSTALL OR ANDROID_APK_RUN)
      add_custom_command(TARGET ${ANDROID_NAME}
	COMMAND adb install -r bin/${ANDROID_NAME}.apk
	WORKING_DIRECTORY "${apk_directory}")
    endif()
  else()
    # Let Ant create the unsigned apk file
    add_custom_command(TARGET ${ANDROID_NAME}
      COMMAND ant debug
      WORKING_DIRECTORY "${apk_directory}")
     
    # Install current version on the device/emulator
    if(ANDROID_APK_INSTALL OR ANDROID_APK_RUN)
      add_custom_command(TARGET ${ANDROID_NAME}
	COMMAND adb install -r bin/${ANDROID_NAME}-debug.apk
	WORKING_DIRECTORY "${apk_directory}")
    endif()
  endif()

  # Start the application
  if(ANDROID_APK_RUN)
    add_custom_command(TARGET ${ANDROID_NAME}
      COMMAND adb shell am start -n ${ANDROID_APK_PACKAGE}/android.app.NativeActivity)
  endif()
endmacro(android_create_apk name apk_directory libs_directory assets_directory)
