use crate::player::Player;
use crate::{MainCamera};
use bevy::input::mouse::MouseMotion;
use bevy::input::ButtonInput;
use bevy::prelude::{BevyError, Component, EventReader, MouseButton, Query, Res, Transform, Vec3, With, Without};
use bevy::window::Window;
use bevy_math::Vec2;

const CAMERA_DISTANCE: f32 = 150.0;

#[derive(Component)]
pub struct CameraController {
    pub(crate) sensitivity: f32,
    pub(crate) pitch: f32,
    pub(crate) yaw: f32,
}

pub fn mouse_look(
    mut windows: Query<&mut Window>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_evr: EventReader<MouseMotion>,
    mut query: Query<(&mut CameraController, &mut Transform)>,
) -> Result<(), BevyError> {
    let mut window = windows.single_mut()?;
    let (mut controller, mut _transform) = query.single_mut()?;

    // Capture cursor when left mouse button is pressed
    if mouse_buttons.just_pressed(MouseButton::Left) {
        window.cursor_options.visible = false;
        window.cursor_options.grab_mode = bevy::window::CursorGrabMode::Locked;
    }

    // Release cursor when left mouse button is released
    if mouse_buttons.just_released(MouseButton::Left) {
        window.cursor_options.visible = true;
        window.cursor_options.grab_mode = bevy::window::CursorGrabMode::None;
    }

    // Calculate mouse delta
    let mut delta = Vec2::ZERO;
    for ev in motion_evr.read() {
        delta += ev.delta;
    }

    // Only rotate if left mouse button is held
    if mouse_buttons.pressed(MouseButton::Left) {
        controller.yaw += delta.x * controller.sensitivity * 0.001;
        controller.pitch += delta.y * controller.sensitivity * 0.001;

        // Keep pitch within -89..89 degrees
        controller.pitch = controller.pitch.clamp(
            -89.0f32.to_radians(),
            89.0f32.to_radians()
        );
    }

    Ok(())
}

pub fn camera_follow(
    drone_query: Query<&Transform, With<Player>>,
    mut camera_query: Query<(&mut Transform, &CameraController), (With<MainCamera>, Without<Player>)>,
) -> Result<(), BevyError> {
    let drone_transform = drone_query.single()?;
    let (mut camera_transform, controller) = camera_query.single_mut()?;

    // Calculate camera offset using spherical coordinates
    let yaw = controller.yaw;
    let pitch = controller.pitch;

    let camera_position = Vec3::new(
        drone_transform.translation.x + yaw.cos() * pitch.cos() * CAMERA_DISTANCE,
        drone_transform.translation.y + pitch.sin() * CAMERA_DISTANCE,
        drone_transform.translation.z + yaw.sin() * pitch.cos() * CAMERA_DISTANCE,
    );

    camera_transform.translation = camera_position;
    camera_transform.look_at(drone_transform.translation, Vec3::Y);
    Ok(())
}

