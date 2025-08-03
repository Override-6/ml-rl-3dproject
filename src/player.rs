use crate::camera_controller::CameraController;
use crate::sensor::ground_sensor::{GroundContact, GroundSensor};
use crate::MainCamera;
use bevy::asset::{Assets, Handle};
use bevy::color::Color;
use bevy::gltf::{Gltf, GltfAssetLabel};
use bevy::input::ButtonInput;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{AssetServer, BevyError, Camera3d, Commands, Component, Entity, GlobalTransform, KeyCode, Mesh, Mesh3d, Quat, Query, Res, ResMut, SceneRoot, Transform, Vec3, With};
use bevy_math::prelude::Cuboid;
use bevy_math::Vec2;
use bevy_rapier3d::dynamics::{AdditionalMassProperties, RigidBody, Velocity};
use bevy_rapier3d::geometry::{Collider, CollisionGroups, Group};
use bevy_rapier3d::prelude::{ActiveEvents, LockedAxes, Sensor};

#[derive(Component)]
pub struct Player();


pub const PLAYER_WIDTH: f32 = 10.0;
pub const PLAYER_HEIGHT: f32 = 10.0;

pub fn setup_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn player entity
    commands
        .spawn((
            Player(),
            Mesh3d(meshes.add(Cuboid::new(PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH))),
            MeshMaterial3d(materials.add(Color::srgb(
                0.7,
                0.2,
                0.5,
            ))),
            LockedAxes::ROTATION_LOCKED,
            RigidBody::Dynamic,
            AdditionalMassProperties::Mass(200.0),
            Collider::cuboid(PLAYER_WIDTH / 2.0, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH / 2.0),
            CollisionGroups::new(Group::GROUP_1, Group::ALL),
            Velocity::zero(),
            Transform::from_xyz(250.0, 40.0, 250.0),
            GroundContact(0)
        ))
        // spawn ground detection
        .with_children(|parent| {
            parent.spawn((
                GroundSensor,
                Collider::cuboid(PLAYER_WIDTH / 2.0, 0.1, PLAYER_WIDTH / 2.0),
                Sensor,
                ActiveEvents::COLLISION_EVENTS,
                Transform::from_xyz(0.0, -(PLAYER_HEIGHT / 2.0), 0.0),
                GlobalTransform::default()
            ));
        });

    // Spawn camera
    commands.spawn((
        CameraController {
            sensitivity: 0.5,
            pitch: 30.0f32.to_radians(),
            yaw: 45.0f32.to_radians(),
        },
        Camera3d::default(),
        Transform::from_xyz(-5.0, 5.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));
}

pub fn move_player(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Velocity, &GroundContact), With<Player>>,
) -> Result<(), BevyError> {
    let (mut velocity, ground_contact) = query.single_mut()?;

    let mut right = 0.0;
    let mut forward = 0.0;

    // Vertical movement
    if keyboard_input.pressed(KeyCode::KeyW) {
        forward -= 1.0;
    } else if keyboard_input.pressed(KeyCode::KeyS) {
        forward += 1.0;
    }

    // Horizontal movement
    if keyboard_input.pressed(KeyCode::KeyA) {
        right -= 1.0;
    } else if keyboard_input.pressed(KeyCode::KeyD) {
        right += 1.0;
    }

    // Apply horizontal speed
    let speed = 200.0;
    velocity.linvel.x = right * speed;
    velocity.linvel.z = forward * speed;

    // Jump
    if keyboard_input.just_pressed(KeyCode::Space) && ground_contact.0 > 0 {
        velocity.linvel.y += 400.0; // Jump impulse
    }

    Ok(())
}
