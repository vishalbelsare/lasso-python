
import typing
import numpy as np


class ArrayType:
    '''Specifies the names for specific arrays

    Enums from this class shall be used as a preferred practice
    instead of the string array names to ensure compatability.
    '''

    # global
    global_timesteps = "timesteps"
    global_kinetic_energy = "global_kinetic_energy"
    global_internal_energy = "global_internal_energy"
    global_total_energy = "global_total_energy"
    global_velocity = "global_velocity"
    # nodes
    node_ids = "node_ids"
    node_coordinates = "node_coordinates"
    node_displacement = "node_displacement"
    node_velocity = "node_velocity"
    node_acceleration = "node_acceleration"
    node_is_alive = "node_is_alive"
    node_temperature = "node_temperature"
    node_heat_flux = "node_heat_flux"
    node_mass_scaling = "node_mass_scaling"
    # solids
    element_solid_part_indexes = "element_solid_part_indexes"
    element_solid_node_indexes = "element_solid_node_indexes"
    element_solid_ids = "element_solid_ids"
    element_solid_thermal_data = "element_solid_thermal_data"
    element_solid_stress = "element_solid_stress"
    element_solid_effective_plastic_strain = \
        "element_solid_effective_plastic_strain"
    element_solid_history_variables = "element_solid_history_variables"
    element_solid_strain = "element_solid_strain"
    element_solid_is_alive = "element_solid_is_alive"
    # tshells
    element_tshell_part_indexes = "element_tshell_part_indexes"
    element_tshell_node_indexes = "element_tshell_node_indexes"
    element_tshell_ids = "element_tshell_ids"
    element_tshell_stress = "element_tshell_stress"
    element_tshell_effective_plastic_strain = \
        "element_tshell_effective_plastic_strain"
    element_tshell_history_variables = \
        "element_tshell_history_variables"
    element_tshell_is_alive = "element_tshell_is_alive"
    element_tshell_strain = "element_tshell_strain"
    # beams
    element_beam_part_indexes = "element_beam_part_indexes"
    element_beam_node_indexes = "element_beam_node_indexes"
    element_beam_ids = "element_beam_ids"
    element_beam_axial_force = "element_beam_axial_force"
    element_beam_shear_force = "element_beam_shear_force"
    element_beam_bending_moment = "element_beam_bending_moment"
    element_beam_torsion_moment = "element_beam_torsion_moment"
    element_beam_shear_stress = "element_beam_shear_stress"
    element_beam_axial_stress = "element_beam_axial_stress"
    element_beam_plastic_strain = "element_beam_plastic_strain"
    element_beam_axial_strain = "element_beam_axial_strain"
    element_beam_history_vars = "element_beam_history_vars"
    element_beam_is_alive = "element_beam_is_alive"
    # shells
    element_shell_part_indexes = "element_shell_part_indexes"
    element_shell_node_indexes = "element_shell_node_indexes"
    element_shell_ids = "element_shell_ids"
    element_shell_stress = "element_shell_stress"
    element_shell_effective_plastic_strain = \
        "element_shell_effective_plastic_strain"
    element_shell_history_vars = "element_shell_history_vars"
    element_shell_bending_moment = "element_shell_bending_moment"
    element_shell_shear_force = "element_shell_shear_force"
    element_shell_normal_force = "element_shell_normal_force"
    element_shell_thickness = "element_shell_thickness"
    element_shell_unknown_variables = "element_shell_unknown_variables"
    element_shell_internal_energy = "element_shell_internal_energy"
    element_shell_strain = "element_shell_strain"
    element_shell_is_alive = "element_shell_is_alive"
    # parts
    part_ids = "part_ids"
    part_titles = "part_titles"
    part_titles_ids = "part_titles_ids"
    part_kinetic_energy = "part_kinetic_energy"
    part_internal_energy = "part_internal_energy"
    part_hourglass_energy = "part_hourglass_energy"
    part_velocity = "part_velocity"
    part_mass = "part_mass"
    # sph
    sph_is_alive = "sph_is_alive"
    sph_radius = "sph_radius"
    sph_pressure = "sph_pressure"
    sph_stress = "sph_stress"
    sph_effective_plastic_strain = "sph_effective_plastic_strain"
    sph_density = "sph_density"
    sph_internal_energy = "sph_internal_energy"
    sph_n_neighbors = "sph_n_neighbors"
    sph_strain = "sph_strain"
    sph_mass = "sph_mass"
    sph_deletion = "sph_deletion"
    # airbag
    airbag_n_active_particles = "airbag_n_active_particles"
    airbag_bag_volume = "airbag_bag_volume"
    airbag_particle_gas_id = "airbag_particle_gas_id"
    airbag_particle_chamber_id = "airbag_particle_chamber_id"
    airbag_particle_leakage = "airbag_particle_leakage"
    airbag_particle_mass = "airbag_particle_mass"
    airbag_particle_radius = "airbag_particle_radius"
    airbag_particle_spin_energy = "airbag_particle_spin_energy"
    airbag_particle_translation_energy = "airbag_particle_translation_energy"
    airbag_particle_nearest_segment_distance = \
        "airbag_particle_nearest_segment_distance"
    airbag_particle_position = "airbag_particle_position"
    airbag_particle_velocity = "airbag_particle_velocity"
    # rigid roads
    rigid_road_displacement = "rigid_road_displacement"
    rigid_road_velocity = "rigid_road_velocity"
    rigid_road_node_ids = "rigid_road_node_ids"
    rigid_road_node_coordinates = "rigid_road_node_coordinates"
    rigid_road_ids = "rigid_road_ids"
    rigid_road_n_segments = "rigid_road_n_segments"
    rigid_road_segment_node_ids = "rigid_road_segment_node_ids"
    rigid_road_segment_road_id = "rigid_road_segment_road_id"
    # rigid body
    rigid_body_coordinates = "rigid_body_coordinates"
    rigid_body_rotation_matrix = "rigid_body_rotation_matrix"
    rigid_body_velocity = "rigid_body_velocity"
    rigid_body_rot_velocity = "rigid_body_rotational_velocity"
    rigid_body_acceleration = "rigid_body_acceleration"
    rigid_body_rot_acceleration = "rigid_body_rotational_acceleration"
    # contact info
    contact_title_ids = "contact_title_ids"
    contact_titles = "contact_titles"
    # ALE
    ale_material_ids = "ale_material_ids"
    # rigid wall
    rigid_wall_force = "rigid_wall_force"
    rigid_wall_position = "rigid_wall_position"

    @staticmethod
    def get_state_array_names() -> typing.List[str]:
        return [
            # global
            ArrayType.global_timesteps,
            ArrayType.global_kinetic_energy,
            ArrayType.global_internal_energy,
            ArrayType.global_total_energy,
            ArrayType.global_velocity,
            # parts
            ArrayType.part_internal_energy,
            ArrayType.part_kinetic_energy,
            ArrayType.part_velocity,
            ArrayType.part_mass,
            ArrayType.part_hourglass_energy,
            # rigid wall
            ArrayType.rigid_wall_force,
            ArrayType.rigid_wall_position,
            # nodes
            ArrayType.node_temperature,
            ArrayType.node_heat_flux,
            ArrayType.node_mass_scaling,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
            ArrayType.node_acceleration,
            # solids
            ArrayType.element_solid_thermal_data,
            ArrayType.element_solid_stress,
            ArrayType.element_solid_effective_plastic_strain,
            ArrayType.element_solid_history_variables,
            ArrayType.element_solid_strain,
            ArrayType.element_solid_is_alive,
            # thick shells
            ArrayType.element_tshell_stress,
            ArrayType.element_tshell_effective_plastic_strain,
            ArrayType.element_tshell_history_variables,
            ArrayType.element_tshell_strain,
            ArrayType.element_tshell_is_alive,
            # beams
            ArrayType.element_beam_axial_force,
            ArrayType.element_beam_shear_force,
            ArrayType.element_beam_bending_moment,
            ArrayType.element_beam_torsion_moment,
            ArrayType.element_beam_shear_stress,
            ArrayType.element_beam_axial_stress,
            ArrayType.element_beam_plastic_strain,
            ArrayType.element_beam_axial_strain,
            ArrayType.element_beam_history_vars,
            ArrayType.element_beam_is_alive,
            # shells
            ArrayType.element_shell_stress,
            ArrayType.element_shell_effective_plastic_strain,
            ArrayType.element_shell_history_vars,
            ArrayType.element_shell_bending_moment,
            ArrayType.element_shell_shear_force,
            ArrayType.element_shell_normal_force,
            ArrayType.element_shell_thickness,
            ArrayType.element_shell_unknown_variables,
            ArrayType.element_shell_internal_energy,
            ArrayType.element_shell_strain,
            ArrayType.element_shell_is_alive,
            # sph
            ArrayType.sph_deletion,
            ArrayType.sph_radius,
            ArrayType.sph_pressure,
            ArrayType.sph_stress,
            ArrayType.sph_effective_plastic_strain,
            ArrayType.sph_density,
            ArrayType.sph_internal_energy,
            ArrayType.sph_n_neighbors,
            ArrayType.sph_strain,
            ArrayType.sph_mass,
            # airbag
            ArrayType.airbag_n_active_particles,
            ArrayType.airbag_bag_volume,
            ArrayType.airbag_particle_gas_id,
            ArrayType.airbag_particle_chamber_id,
            ArrayType.airbag_particle_leakage,
            ArrayType.airbag_particle_mass,
            ArrayType.airbag_particle_radius,
            ArrayType.airbag_particle_spin_energy,
            ArrayType.airbag_particle_translation_energy,
            ArrayType.airbag_particle_nearest_segment_distance,
            ArrayType.airbag_particle_position,
            ArrayType.airbag_particle_velocity,
            # rigid road
            ArrayType.rigid_road_displacement,
            ArrayType.rigid_road_velocity,
            # rigid body
            ArrayType.rigid_body_coordinates,
            ArrayType.rigid_body_rotation_matrix,
            ArrayType.rigid_body_velocity,
            ArrayType.rigid_body_rot_velocity,
            ArrayType.rigid_body_acceleration,
            ArrayType.rigid_body_rot_acceleration,
        ]
