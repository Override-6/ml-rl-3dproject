use bevy::input::ButtonInput;
use bevy::prelude::{BevyError, Component, KeyCode, Query, Res, Time, Transform, Vec3};
use bevy_rapier3d::dynamics::Velocity;

#[derive(Component)]
struct Drone {
    max_speed: f32,
    acceleration: f32,
    max_thrust: f32,
    rotation_speed: f32,
}

fn drone_movement(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&Drone, &mut Velocity, &Transform)>,
    time: Res<Time>,
) -> Result<(), BevyError> {
    let (drone, mut velocity, transform) = query.single_mut()?;
    let mut thrust = Vec3::ZERO;

    // Movement controls
    if keyboard_input.pressed(KeyCode::KeyW) {
        thrust += *transform.forward();
    }
    if keyboard_input.pressed(KeyCode::KeyA) {
        thrust += *transform.left();
    }
    if keyboard_input.pressed(KeyCode::KeyS) {
        thrust += *transform.back();
    }
    if keyboard_input.pressed(KeyCode::KeyD) {
        thrust += *transform.right();
    }
    // Altitude controls
    if keyboard_input.pressed(KeyCode::Space) {
        thrust += *transform.up();
    }
    if keyboard_input.pressed(KeyCode::ShiftLeft) {
        thrust += *transform.down();
    }

    if thrust != Vec3::ZERO {
        thrust = thrust.normalize() * drone.max_thrust;
        velocity.linvel += thrust * time.delta_secs() * drone.acceleration;
    }

    // Apply forces with air resistance considered
    // let speed = velocity.linvel.length();
    // let effective_thrust = thrust * (1.0 - speed / drone.max_speed);
    //
    // if thrust != Vec3::ZERO {
    //     velocity.linvel += effective_thrust * time.delta_secs();
    // }

    // Rotation controls
    let mut rotation = Vec3::ZERO;
    if keyboard_input.pressed(KeyCode::ArrowLeft) {
        rotation.y += drone.rotation_speed * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::ArrowRight) {
        rotation.y -= drone.rotation_speed * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::ArrowUp) {
        rotation.x -= drone.rotation_speed * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::ArrowDown) {
        rotation.x += drone.rotation_speed * time.delta_secs();
    }
    velocity.angvel += rotation;

    Ok(())
}
