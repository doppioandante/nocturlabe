project('nocturlabe', 'cpp', default_options : ['cpp_std=c++17'])

cpp = meson.get_compiler('cpp')
sundials_ida = cpp.find_library('sundials_ida', required: true)
sundials_nvecserial = cpp.find_library('sundials_nvecserial', required: true)
sundials_nvecmanyvector = cpp.find_library('sundials_nvecmanyvector', required: true)
sundials_sunlinsolklu = cpp.find_library('sundials_sunlinsolklu', required: true)
klu = cpp.find_library('klu', required: true)
cppad = dependency('cppad', required: true)

install_headers('nocturlabe.hpp', subdir: 'nocturlabe.hpp')
nocturlabe_dep = declare_dependency(
  include_directories : include_directories('/usr/include/suitesparse/', '.'),
  dependencies : [sundials_ida, sundials_nvecserial, sundials_nvecmanyvector, klu, sundials_sunlinsolklu, cppad],
)
