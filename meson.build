project('astrolabio', 'cpp', default_options : ['cpp_std=c++17'])

cpp = meson.get_compiler('cpp')
sundials_ida = cpp.find_library('sundials_ida', required: true)
sundials_nvecserial = cpp.find_library('sundials_nvecserial', required: true)
sundials_nvecmanyvector = cpp.find_library('sundials_nvecmanyvector', required: true)
sundials_sunlinsolklu = cpp.find_library('sundials_sunlinsolklu', required: true)
klu = cpp.find_library('klu', required: true)
cppad = dependency('cppad', required: true)

install_headers(files('include/astrolabio/astrolabio.hpp'), subdir: 'astrolabio')
astrolabio_dep = declare_dependency(
  include_directories : include_directories('/usr/include/suitesparse/', './include'),
  dependencies : [sundials_ida, sundials_nvecserial, sundials_nvecmanyvector, klu, sundials_sunlinsolklu, cppad],
)

examples = [
    'examples/horizontal_particle.cpp',
]

foreach e : examples
    executable(e.split('/')[1].replace('.cpp', ''), files(e), dependencies : astrolabio_dep)
endforeach
