use std::f32::consts::PI;
use crate::ai::input::PulseInput;
use crate::ai::model_control_pipeline::SimulationPlayersInputs;
use crate::player::{Player, PLAYER_JUMP_SPEED, PLAYER_SPEED, PLAYER_TURN_SPEED};
use crate::sensor::ground_sensor::GroundContact;
use crate::sensor::player_vibrissae::PlayerVibrissae;
use crate::simulation::DELTA_TIME;
use bevy::app::AppExit;
use bevy::prelude::{BevyError, Component, EventWriter, Query, Res, Transform, Vec3, With};
use bevy_rapier3d::prelude::Velocity;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicBool, Ordering};
use rand::Rng;

#[derive(Component)]
pub struct AIPlayer;

pub fn follow_all_script(
    mut player_query: Query<
        (
            &mut Velocity,
            &Transform,
            &mut GroundContact,
            &mut Player,
            &mut PlayerVibrissae,
        ),
        With<AIPlayer>,
    >,
    sim: Res<SimulationPlayersInputs>,
    mut app_exit: EventWriter<AppExit>,
) -> bevy::prelude::Result<(), BevyError> {
    let should_exit: AtomicBool = AtomicBool::default();
    player_query.par_iter_mut().for_each(
        |(mut velocity, transform, mut ground_contact, mut player, mut vibrissae)| {
            if player.freeze {
                return;
            }

            let should_stop = follow_script(
                velocity.deref_mut(),
                transform,
                ground_contact.deref_mut(),
                player.deref_mut(),
                &sim,
                &mut vibrissae
            );
            if should_stop {
                should_exit.store(should_stop, Ordering::Relaxed);
            }
        },
    );

    if should_exit.load(Ordering::Relaxed) {
        app_exit.write(AppExit::Success);
    }

    Ok(())
}

/// returns true if app should stop.
fn follow_script(
    velocity: &mut Velocity,
    transform: &Transform,
    ground_contact: &mut GroundContact,
    ai_player: &mut Player,
    inputs: &Res<SimulationPlayersInputs>,
    vibrissae: &mut PlayerVibrissae,
) -> bool {

    let input = &inputs.inputs[ai_player.id];

    let pulse_set = input.pulse;
    let laser_dir_yaw = input.laser_dir - 0.5;
    let yaw: f32 = laser_dir_yaw * std::f32::consts::TAU;

    let centered = Vec3::new(
        yaw.sin(),
        0.0,
        -yaw.cos(),
    );

    // change direction of the directional laser based on model input
    vibrissae.get_directional_laser().direction = centered;

    let mut move_input = Vec3::ZERO;

    // Forward/Backward
    if pulse_set & PulseInput::Forward != 0 {
        move_input.z -= 1.0;
    } else if pulse_set & PulseInput::Backward != 0 {
        move_input.z += 1.0;
    }

    // Left/Right
    if pulse_set & PulseInput::Left != 0 {
        move_input.x -= 1.0;
    } else if pulse_set & PulseInput::Right != 0 {
        move_input.x += 1.0;
    }

    // Normalize movement input to prevent diagonal speed boost
    if move_input.length_squared() > 0.0 {
        move_input = move_input.normalize();
    }

    let speed = PLAYER_SPEED;

    // Apply movement relative to player's facing direction
    let rotated_input = transform.rotation * move_input;
    velocity.linvel.x = rotated_input.x * speed;
    velocity.linvel.z = rotated_input.z * speed;

    // Yaw rotation (turn left or right)
    let mut yaw = 0.0;
    if pulse_set & PulseInput::TurnLeft != 0 {
        yaw += PLAYER_TURN_SPEED;
    }
    if pulse_set & PulseInput::TurnRight != 0 {
        yaw -= PLAYER_TURN_SPEED;
    }

    velocity.angvel.y = yaw * DELTA_TIME;

    // print!("{:?} ", velocity.linvel.y);

    // Jump
    if pulse_set & PulseInput::Jump != 0 {
        if ground_contact.0 != 0 {
            velocity.linvel.y += PLAYER_JUMP_SPEED;
            ground_contact.0 = 0;
        } else {
            // error!("Received jump instruction while not on ground!")
        }
    }

    ai_player.is_interacting = pulse_set & PulseInput::Interact != 0;

    false
}
