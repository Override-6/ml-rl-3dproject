use crate::ai::input::Input;
use crate::ai::model_control_pipeline::SimulationPlayersInputs;
use crate::player::{PLAYER_SPEED, PLAYER_TURN_SPEED, Player};
use crate::sensor::ground_sensor::GroundContact;
use crate::simulation::DELTA_TIME;
use bevy::app::AppExit;
use bevy::prelude::{BevyError, Component, EventWriter, Query, Res, Transform, Vec3, With};
use bevy::time::Time;
use bevy_rapier3d::prelude::Velocity;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Component)]
pub struct AIPlayer;

pub fn follow_all_script(
    mut player_query: Query<
        (
            &mut Velocity,
            &Transform,
            &GroundContact,
            &mut Player,
        ),
        With<AIPlayer>,
    >,
    sim: Res<SimulationPlayersInputs>,
    mut app_exit: EventWriter<AppExit>,
) -> bevy::prelude::Result<(), BevyError> {
    let should_exit: AtomicBool = AtomicBool::default();
    player_query.iter_mut().for_each(
        |(mut velocity, transform, ground_contact, mut player)| {
            if player.freeze {
                return;
            }

            let should_stop = follow_script(
                velocity.deref_mut(),
                transform,
                ground_contact,
                player.deref_mut(),
                &sim,
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
    ground_contact: &GroundContact,
    ai_player: &mut Player,
    sim: &Res<SimulationPlayersInputs>,
) -> bool {
    let input_set = sim.inputs[ai_player.id];

    let mut move_input = Vec3::ZERO;

    // Forward/Backward
    if input_set & Input::Forward != 0 {
        move_input.z -= 1.0;
    } else if input_set & Input::Backward != 0 {
        move_input.z += 1.0;
    }

    // Left/Right
    if input_set & Input::Left != 0 {
        move_input.x -= 1.0;
    } else if input_set & Input::Right != 0 {
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
    if input_set & Input::TurnLeft != 0 {
        yaw += PLAYER_TURN_SPEED;
    }
    if input_set & Input::TurnRight != 0 {
        yaw -= PLAYER_TURN_SPEED;
    }

    velocity.angvel.y = yaw * DELTA_TIME;

    // Jump
    if input_set & Input::Jump != 0 {
        if ground_contact.0 != 0 {
            velocity.linvel.y += 400.0;
        } else {
            // error!("Received jump instruction while not on ground!")
        }
    }

    false
}
