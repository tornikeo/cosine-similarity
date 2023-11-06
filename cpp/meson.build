project('cosine','cpp')
libnpydep = include_directories('include')
eigen_proj = subproject('eigen')
eigen_dep = eigen_proj.get_variable('eigen_dep')

executable('main', 'main.cpp', 
    include_directories : libnpydep,
    dependencies: eigen_dep,
    install: true
)
