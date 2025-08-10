use crate::ai::input::Input;
use crate::human::player::PLAYER_TURN_SPEED;
use crate::sensor::ground_sensor::GroundContact;
use crate::simulation::{Simulation, DELTA_TIME};
use bevy::app::AppExit;
use bevy::log::error;
use bevy::prelude::{
    BevyError, Component, EventWriter, Query, Res, Transform, Vec3, With,
};
use bevy_rapier3d::prelude::Velocity;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use bevy::time::Time;

pub struct AIPlayerId(pub usize);

#[derive(Component)]
pub struct AIPlayer {
    script_progress: usize,
    pub id: AIPlayerId,
}

impl AIPlayer {
    pub fn new(id: AIPlayerId) -> Self {
        Self {
            id,
            script_progress: 0,
        }
    }
}

pub fn follow_all_script(
    mut player_query: Query<
        (&mut Velocity, &Transform, &GroundContact, &mut AIPlayer),
        With<AIPlayer>,
    >,
    sim: Res<Simulation>,
    mut app_exit: EventWriter<AppExit>,
    time: Res<Time>
) -> bevy::prelude::Result<(), BevyError> {

    let should_exit: AtomicBool = AtomicBool::default();
    player_query
        .par_iter_mut()
        .for_each(|(mut velocity, transform, ground_contact, mut ai_player)| {
            let should_stop = follow_script(velocity.deref_mut(), transform, ground_contact, ai_player.deref_mut(), &sim);
            if should_stop {
                should_exit.store(should_stop, Ordering::Relaxed);
            }
        });

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
    ai_player: &mut AIPlayer,
    sim: &Res<Simulation>,
) -> bool {
    let index = ai_player.script_progress;
    ai_player.script_progress += 1;

    let script = match sim.deref() {
        Simulation::Simulation { script, .. } => script.clone(),
        _ => unreachable!("No AI Entity should live if not in Simulation Mode!"),
    };

    if index >= script.inputs.len() {
        return true; // abort once script finished
    }

    let input_set = script.inputs[index];

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

    let speed = 200.0;

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
        if ground_contact.0 == 0 {
            error!("Received jump instruction while not on ground!")
        } else {
            velocity.linvel.y += 400.0;
        }
    }

    false
}
