use crate::player::Player;
use bevy::input::mouse::MouseMotion;
use bevy::input::ButtonInput;
use bevy::prelude::{BevyError, Camera3d, Commands, Component, EventReader, MouseButton, Query, Res, Transform, Vec3, With, Without};
use bevy::window::Window;
use bevy_math::Vec2;

const CAMERA_DISTANCE: f32 = 150.0;

#[derive(Component)]
pub struct MainCamera;

#[derive(Component)]
pub struct CameraController {
    pub(crate) sensitivity: f32,
    pub(crate) pitch: f32,
    pub(crate) yaw: f32,
}

pub fn spawn_camera_controller(mut commands: Commands) {
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

pub fn mouse_look(
    mut windows: Query<&mut Window>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_evr: EventReader<MouseMotion>,
    mut query: Query<&mut CameraController>,
) -> Result<(), BevyError> {
    let mut window = windows.single_mut()?;
    let mut controller = query.single_mut()?;

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
        controller.yaw += delta.x * controller.sensitivity;
        controller.pitch += delta.y * controller.sensitivity;

        // Keep pitch within -89..89 degrees
        controller.pitch = controller
            .pitch
            .clamp(-89.0f32.to_radians(), 89.0f32.to_radians());
    }

    Ok(())
}

pub fn camera_follow(
    player_query: Query<&Transform, With<Player>>,
    mut camera_query: Query<
        (&mut Transform, &CameraController),
        (With<MainCamera>, Without<Player>),
    >,
) -> Result<(), BevyError> {
    // Attach to the first player
    let player_transform = player_query.iter().next().ok_or_else(|| BevyError::from("No player found."))?;

    let (mut camera_transform, controller) = camera_query.single_mut()?;

    let yaw = controller.yaw;
    let pitch = controller.pitch;

    let camera_position = Vec3::new(
        player_transform.translation.x + yaw.cos() * pitch.cos() * CAMERA_DISTANCE,
        player_transform.translation.y + pitch.sin() * CAMERA_DISTANCE,
        player_transform.translation.z + yaw.sin() * pitch.cos() * CAMERA_DISTANCE,
    );

    camera_transform.translation = camera_position;
    camera_transform.look_at(player_transform.translation, Vec3::Y);

    Ok(())
}
