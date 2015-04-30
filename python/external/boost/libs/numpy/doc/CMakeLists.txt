project(doc-html)
file(GLOB NUMPY_DOC_DEPS conf.py *.rst)
message( STATUS "NUMPY_DOC_DEPS=${NUMPY_DOC_DEPS}" )

# add_custom_target(doc-html make -C ${PROJECT_SOURCE_DIR} html BUILDDIR=${PROJECT_BINARY_DIR}/_build)
#
# this custom target is a cross-platform python/sphinx way to
# replicate what the above make command is doing.
#
add_custom_target(doc-html
  ${SPHINX_BUILD}
     -b html 
     -c ${CMAKE_CURRENT_SOURCE_DIR}
     -d .doctrees
     ${CMAKE_CURRENT_SOURCE_DIR}
     html
  
  DEPENDS ${NUMPY_DOC_DEPS}
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Generating HTML Documentation"
)

SET_PROPERTY(TARGET doc-html PROPERTY FOLDER "doc")

install(DIRECTORY ${PROJECT_BINARY_DIR}/html
  DESTINATION share/doc/libboost_numpy 
  OPTIONAL
  )

if (PDFLATEX_COMPILER)
  project(doc-pdf)

  add_custom_target(doc-pdf
    ${SPHINX_BUILD}
      -b latex 
      -c ${CMAKE_CURRENT_SOURCE_DIR}
      -d .doctrees
      ${CMAKE_CURRENT_SOURCE_DIR}
      latex
    
    COMMAND ${PDFLATEX_COMPILER} --include-directory=latex --output-directory=latex latex/BoostNumPy.tex 
    COMMAND ${PDFLATEX_COMPILER} --include-directory=latex --output-directory=latex latex/BoostNumPy.tex 

    DEPENDS ${NUMPY_DOC_DEPS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Generating Latex-pdf Documentation"
    )

  SET_PROPERTY(TARGET doc-pdf PROPERTY FOLDER "doc")

  install(FILES ${PROJECT_BINARY_DIR}/latex/BoostNumPy.pdf
    DESTINATION share/doc/libboost_numpy
    OPTIONAL
    )

endif()
