"""
    BuoyancyScheme
An abstract type for buoyancy schemes.
"""
abstract type BuoyancyScheme end

""" 
    BuoyDyn
An abstract type for buoyancy schemes based on pressure and buoyancy.
"""
abstract type BuoyDyn <: BuoyancyScheme end

"""
    BuoyThermo
A type for the buoyancy scheme based on thermodynamic variables.
"""
struct BuoyThermo <: BuoyancyScheme end

"""
    BuoyDynNew
A type for the new buoyancy scheme based on the buoyancy formulation in Tailleux & Dubos (2024).
"""
struct BuoyDynNew <: BuoyDyn end

"""
    BuoyDynOld
A type for the old buoyancy scheme based on the traditional buoyancy formulation
"""
struct BuoyDynOld <: BuoyDyn end