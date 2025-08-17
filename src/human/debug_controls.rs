use crate::simulation::{SimulationConfig, SimulationState};
use bevy::input::ButtonInput;
use bevy::prelude::{KeyCode, Res, ResMut};


pub fn debug_controls(
    kb: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<SimulationState>,
) {
    if kb.just_pressed(KeyCode::Numpad1) {
        sim.debug.print_all_lasers = ! sim.debug.print_all_lasers;
    }
}
