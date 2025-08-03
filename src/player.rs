use crate::camera_controller::{CameraController, MainCamera};
use crate::sensor::ground_sensor::{GroundContact, GroundSensor};
use bevy::asset::Assets;
use bevy::color::palettes::basic::{BLUE, RED};
use bevy::color::Color;
use bevy::input::ButtonInput;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{BevyError, Camera3d, Commands, Component, Entity, Gizmos, GlobalTransform, KeyCode, Mesh, Mesh3d, MeshRayCast, MeshRayCastSettings, Quat, Query, Res, ResMut, Time, Transform, Vec3, With, Without};
use bevy_math::prelude::Cuboid;
use bevy_math::{Dir3, Ray3d};
use bevy_rapier3d::dynamics::{AdditionalMassProperties, CoefficientCombineRule, RigidBody, Velocity};
use bevy_rapier3d::geometry::{Collider, CollisionGroups, Group};
use bevy_rapier3d::prelude::{ActiveEvents, Friction, LockedAxes, Sensor};

#[derive(Component)]
pub struct Player();

pub const PLAYER_WIDTH: f32 = 10.0;
pub const PLAYER_HEIGHT: f32 = 10.0;

pub const PLAYER_TURN_SPEED: f32 = 100.0;

pub const RAYCASTS_DEBUG_DISPLAY_DISTANCE: f32 = 3000.0;

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
            MeshMaterial3d(materials.add(Color::srgb(0.7, 0.2, 0.5))),
            LockedAxes::ROTATION_LOCKED ^ LockedAxes::ROTATION_LOCKED_Y,
            RigidBody::Dynamic,
            AdditionalMassProperties::Mass(200.0),
            Collider::cuboid(PLAYER_WIDTH / 2.0, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH / 2.0),
            CollisionGroups::new(Group::GROUP_1, Group::ALL),
            Velocity::zero(),
            Friction {
                coefficient: 0.0,
                combine_rule: CoefficientCombineRule::Min,
            },
            GroundContact(0),
        ))
        .with_children(|parent| {
            // spawn ground detection
            parent.spawn((
                GroundSensor,
                Collider::cuboid(PLAYER_WIDTH / 2.0, 0.1, PLAYER_WIDTH / 2.0),
                Sensor,
                ActiveEvents::COLLISION_EVENTS,
                Transform::from_xyz(0.0, -(PLAYER_HEIGHT / 2.0), 0.0),
                GlobalTransform::default(),
            ));
        });

        // Spawn camera
        commands.spawn((
            CameraController {
                sensitivity: 0.005,
                pitch: 30.0f32.to_radians(),
                yaw: 45.0f32.to_radians(),
            },
            Camera3d::default(),
            Transform::from_xyz(-50.0, 250.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
            MainCamera,
        ));
}

pub fn move_player(
    time: Res<Time>,
    kb: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Velocity, &GroundContact), (With<Player>, Without<CameraController>)>,
) -> Result<(), BevyError> {
    let (mut velocity, ground_contact) = query.single_mut()?;

    let mut right = 0.0;
    let mut forward = 0.0;

    // Vertical movement
    if kb.pressed(KeyCode::KeyW) {
        forward -= 1.0;
    } else if kb.pressed(KeyCode::KeyS) {
        forward += 1.0;
    }

    // Horizontal movement
    if kb.pressed(KeyCode::KeyA) {
        right -= 1.0;
    } else if kb.pressed(KeyCode::KeyD) {
        right += 1.0;
    }

    let speed = 200.0;
    velocity.linvel.x = right * speed;
    velocity.linvel.z = forward * speed;

    // Jump
    if kb.just_pressed(KeyCode::Space) && ground_contact.0 > 0 {
        velocity.linvel.y += 400.0;
    }

    let mut yaw = 0.0;

    if kb.pressed(KeyCode::ArrowRight) {
        yaw -= PLAYER_TURN_SPEED;
    }
    if kb.pressed(KeyCode::ArrowLeft) {
        yaw += PLAYER_TURN_SPEED;
    }

    velocity.angvel.y = yaw * time.delta_secs();


    Ok(())
}

pub fn player_raycast_update(
    query: Query<(Entity, &GlobalTransform), With<Player>>,
    mut gizmos: Gizmos,
    mut mesh_ray_cast: MeshRayCast,
) -> Result<(), BevyError> {
    let (player_ent, player_gt) = query.single()?;

    let origin = player_gt.translation();
    let direction = player_gt.rotation();

    let direction = direction * Vec3::NEG_Z;

    let filter = move |entity| entity != player_ent;
    let settings = MeshRayCastSettings::default()
        .with_filter(&filter)
        .always_early_exit();

    let ray = Ray3d::new(origin, Dir3::try_from(direction)?);

    let results = mesh_ray_cast.cast_ray(ray, &settings);

    if let Some((entity, hit)) = results.first() {
        println!(
            "Player is facing mesh hit at entity {:?}, distance {:.1}",
            entity, hit.distance
        );
        gizmos.line(
            origin,
            origin + direction * RAYCASTS_DEBUG_DISPLAY_DISTANCE,
            RED,
        );

        gizmos.sphere(hit.point, 3.0, RED);
    } else {
        gizmos.line(
            origin,
            origin + direction * RAYCASTS_DEBUG_DISPLAY_DISTANCE,
            BLUE,
        );
    }

    Ok(())
}
