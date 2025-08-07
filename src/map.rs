use bevy::asset::Assets;
use bevy::color::Color;
use bevy::pbr::{DirectionalLight, MeshMaterial3d, StandardMaterial};
use bevy::prelude::{default, Commands, Mesh, Mesh3d, Name, ResMut, Transform, Vec2, Vec3};
use bevy_math::prelude::{Cuboid, Plane3d};
use bevy_rapier3d::dynamics::RigidBody;
use bevy_rapier3d::geometry::{ActiveEvents, Collider, CollisionGroups, Friction, Group, Sensor};
use crate::objective::TriggerZone;

pub fn setup_map(mut commands: Commands,
                 mut meshes: ResMut<Assets<Mesh>>,
                 mut materials: ResMut<Assets<StandardMaterial>>) {
    // Plane
    commands.spawn((
        Mesh3d::from(meshes.add(Plane3d::new(Vec3::Y, Vec2::new(500.0, 500.0)))),
        MeshMaterial3d::from(materials.add(Color::srgb(1.0, 1.0, 1.0))),
        Collider::halfspace(Vec3::Y).unwrap(),
        CollisionGroups::new(Group::GROUP_2, Group::ALL),
        Friction::coefficient(0.5),
        RigidBody::Fixed,
        Name::new("Ground")
    ));

    // Spawn Cube Obstacle
    commands.spawn((
        Mesh3d::from(meshes.add(Cuboid::new(50.0, 50.0, 50.0))),
        MeshMaterial3d::from(materials.add(Color::srgb(0.5, 0.5, 0.1))),
        Transform::from_xyz(250.0, 10.0, 250.0),
        Collider::cuboid(25.0, 25.0, 25.0),
        RigidBody::Fixed,
        CollisionGroups::new(Group::GROUP_2, Group::ALL),
        Friction::coefficient(0.8),
    ));

    // Spawn objective
    commands.spawn((
        Mesh3d::from(meshes.add(Cuboid::new(50.0, 50.0, 50.0))),
        MeshMaterial3d::from(materials.add(Color::srgba(0.0, 1.0, 0.0, 0.3))),
        Transform::from_xyz(-250.0, 10.0, -250.0),
        RigidBody::Fixed,
        Collider::cuboid(25.0, 25.0, 25.0),
        CollisionGroups::new(Group::GROUP_2, Group::ALL),
        Sensor,
        ActiveEvents::COLLISION_EVENTS,
        TriggerZone,
    ));

    // Spawn two point lights
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(20.0, 20.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-20.0, 20.0, -20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}