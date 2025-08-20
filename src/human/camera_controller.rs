use crate::player::Player;
use crate::sensor::objective::Objective;
use bevy::input::ButtonInput;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::{
    BevyError, Camera3d, Commands, Component, EventReader, KeyCode, MouseButton, Quat, Query, Res,
    Time, Transform, Vec3, Vec3Swizzles, Window, With, Without, warn,
};
use bevy_math::Vec2;
use std::cmp::PartialEq;

const DEFAULT_CAMERA_DISTANCE: f32 = 150.0;

#[derive(Component)]
pub struct MainCamera;

#[derive(PartialEq, Eq)]
pub enum CameraMode {
    FollowPlayer,
    FollowObjective,
    Free,
}

#[derive(Component)]
pub struct CameraController {
    pub sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub distance: f32,
    pub speed: f32,
    pub mode: CameraMode,
}

pub fn spawn_camera_controller(mut commands: Commands) {
    commands.spawn((
        MainCamera,
        CameraController {
            sensitivity: 0.0005,
            pitch: 30.0f32.to_radians(),
            yaw: 45.0f32.to_radians(),
            distance: DEFAULT_CAMERA_DISTANCE,
            speed: 200.0,
            mode: CameraMode::FollowPlayer,
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

    // right-click cycles camera mode
    if mouse_buttons.just_pressed(MouseButton::Right) {
        controller.mode = match controller.mode {
            CameraMode::FollowPlayer => CameraMode::FollowObjective,
            CameraMode::FollowObjective => CameraMode::Free,
            CameraMode::Free => CameraMode::FollowPlayer,
        };
    }

    // Sum mouse motion for this frame
    let mut delta = Vec2::ZERO;
    for ev in motion_evr.read() {
        delta += ev.delta;
    }

    // Only rotate while left mouse button is held (change if you want always-on)
    if mouse_buttons.pressed(MouseButton::Left) {
        // Horizontal: yaw (left/right)
        controller.yaw += delta.x * controller.sensitivity;
        // Vertical: pitch (invert sign so moving mouse up pitches camera up)
        controller.pitch -= delta.y * controller.sensitivity;

        // Keep pitch within -89..89 degrees
        controller.pitch = controller
            .pitch
            .clamp(-89.0f32.to_radians(), 89.0f32.to_radians());
    }

    Ok(())
}

pub fn mouse_zoom(
    mut evr_scroll: EventReader<MouseWheel>,
    mut query: Query<&mut CameraController>,
) {
    if let Ok(mut controller) = query.single_mut() {
        for ev in evr_scroll.read() {
            if controller.mode == CameraMode::Free {
                controller.speed = (controller.speed + ev.y * 3.0).max(1.0);
            } else {
                controller.distance = (controller.distance - ev.y * 3.0).max(1.0);
            }
        }
    }
}

pub fn camera_follow(
    player_query: Query<&Transform, With<Player>>,
    objective_query: Query<&Transform, With<Objective>>,
    time: Res<Time>,
    kb: Res<ButtonInput<KeyCode>>,
    mut camera_query: Query<
        (&mut Transform, &CameraController),
        (With<MainCamera>, Without<Player>, Without<Objective>),
    >,
) {
    let (mut camera_transform, controller) = match camera_query.single_mut() {
        Ok(pair) => pair,
        Err(e) => {
            warn!("camera_follow: no camera found: {}", e);
            return;
        }
    };

    match controller.mode {
        CameraMode::FollowPlayer => {
            if let Ok(player_transform) = player_query.single() {
                // Build rotation from yaw (Y) and pitch (X)
                let rot = Quat::from_axis_angle(Vec3::Y, controller.yaw)
                    * Quat::from_axis_angle(Vec3::X, controller.pitch);

                // Place the camera at target + rotated offset (distance along local Z)
                let offset = rot.mul_vec3(Vec3::new(0.0, 0.0, controller.distance));
                camera_transform.translation = player_transform.translation + offset;

                // Look at the player
                camera_transform.look_at(player_transform.translation, Vec3::Y);
            }
        }

        CameraMode::FollowObjective => {
            if let Ok(objective_transform) = objective_query.single() {
                let rot = Quat::from_axis_angle(Vec3::Y, -controller.yaw)
                    * Quat::from_axis_angle(Vec3::X, controller.pitch);

                let offset = rot.mul_vec3(Vec3::new(0.0, 0.0, controller.distance));
                camera_transform.translation = objective_transform.translation + offset;

                camera_transform.look_at(objective_transform.translation, Vec3::Y);
            }
        }

        CameraMode::Free => {
            // Use yaw / pitch directly to set camera rotation
            let rot = Quat::from_axis_angle(Vec3::Y, -controller.yaw)
                * Quat::from_axis_angle(Vec3::X, controller.pitch);
            camera_transform.rotation = rot;

            let forward = camera_transform.forward().xyz();
            let right = camera_transform.right().xyz();
            let up = camera_transform.up().xyz();

            let mut movement = Vec3::ZERO;
            if kb.pressed(KeyCode::KeyW) {
                movement += forward;
            }
            if kb.pressed(KeyCode::KeyS) {
                movement -= forward;
            }
            if kb.pressed(KeyCode::KeyD) {
                movement += right;
            }
            if kb.pressed(KeyCode::KeyA) {
                movement -= right;
            }
            if kb.pressed(KeyCode::Space) {
                movement += up;
            }
            if kb.pressed(KeyCode::ShiftLeft) {
                movement -= up;
            }

            if movement.length_squared() > 0.0 {
                movement = movement.normalize();
                camera_transform.translation += movement * controller.speed * time.delta_secs();
            }
        }
    }
}
