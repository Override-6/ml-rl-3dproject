use bevy::input::ButtonInput;
use bevy::prelude::{Commands, KeyCode, Res};

pub fn update_observer_controls(mut kb: Res<ButtonInput<KeyCode>>, mut commands: Commands) {
    if kb.just_pressed(KeyCode::F3) {
    }
}