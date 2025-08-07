use std::ops::BitAnd;
use crate::ai::input::{Input, InputSet};
use crate::ai::input_recorder::InputRecorder;
use crate::ai::script::Script;
use crate::component::player_character::{PLAYER_HEIGHT, PLAYER_WIDTH};
use crate::human::camera_controller::{CameraController, MainCamera};
use crate::human::player::PLAYER_TURN_SPEED;
use crate::player::Player;
use crate::sensor::ground_sensor::GroundContact;
use bevy::app::AppExit;
use bevy::input::ButtonInput;
use bevy::log::error;
use bevy::prelude::{
    BevyError, Camera3d, Commands, Component, EventWriter, InheritedVisibility, KeyCode, Query,
    Res, ResMut, Time, Transform, Vec3, With,
};
use bevy_rapier3d::dynamics::{
    AdditionalMassProperties, CoefficientCombineRule, RigidBody, Velocity,
};
use bevy_rapier3d::geometry::{Collider, CollisionGroups, Friction, Group};

#[derive(Component)]
pub struct AIPlayer {
    script: Script,
    script_progress: usize,
}

// TODO can be factorized with human::player.rs
pub fn spawn_ai_player(mut commands: Commands, script: Res<Script>) {
    commands.spawn((
        AIPlayer {
            script: script.clone(),
            script_progress: 0,
        },
        Player,
        Transform::default(),
        Velocity::default(),
        InheritedVisibility::VISIBLE,
        GroundContact(0),
        RigidBody::Dynamic,
        AdditionalMassProperties::Mass(200.0),
        Collider::cuboid(PLAYER_WIDTH / 2.0, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH / 2.0),
        CollisionGroups::new(Group::GROUP_1, Group::ALL),
        Friction {
            coefficient: 0.0,
            combine_rule: CoefficientCombineRule::Min,
        },
    ));

    // Spawn camera
    commands.spawn((
        MainCamera,
        CameraController {
            sensitivity: 0.0005,
            pitch: 30.0f32.to_radians(),
            yaw: 45.0f32.to_radians(),
        },
        Camera3d::default(),
        Transform::from_xyz(-50.0, 250.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

pub fn follow_script(
    time: Res<Time>,
    mut player_query: Query<(&mut Velocity, &Transform, &GroundContact, &mut AIPlayer), With<AIPlayer>>,
    mut input_recorder: ResMut<InputRecorder<100>>,
    mut app_exit: EventWriter<AppExit>,
) -> bevy::prelude::Result<(), BevyError> {
    let (mut velocity, transform, ground_contact, mut ai_player) = player_query.single_mut()?;

    let index = ai_player.script_progress;
    ai_player.script_progress += 1;

    let script = &mut ai_player.script;
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

    velocity.angvel.y = yaw * time.delta_secs();

    // Jump
    if input_set & Input::Jump != 0 {
        if ground_contact.0 == 0 {
            error!("Received jump instruction while not on ground!")
        } else {
            velocity.linvel.y += 400.0;
        }
    }

    println!("{:?}", input_set);

    if input_recorder.is_full() {
        app_exit.write(AppExit::Success);
        return Ok(());
    }

    Ok(())
}
