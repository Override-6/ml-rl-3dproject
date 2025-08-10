use crate::sensor::objective::TriggerZone;
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::pbr::{DirectionalLight, MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Commands, Component, Mesh, Mesh3d, Name, ResMut, Transform, Vec2, Vec3, default,
};
use bevy_math::prelude::{Cuboid, Plane3d};
use bevy_rapier3d::dynamics::RigidBody;
use bevy_rapier3d::geometry::{ActiveEvents, Collider, CollisionGroups, Friction, Group, Sensor};

#[derive(Debug, Component, Clone, Copy)]
pub enum ComponentType {
    Ground,
    Object,
    Objective,
    Unknown,
}

pub fn setup_map(
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    // Plane
    let mut plane = commands.spawn((
        Collider::halfspace(Vec3::Y).unwrap(),
        CollisionGroups::new(Group::GROUP_2, Group::ALL),
        Friction::coefficient(0.5),
        RigidBody::Fixed,
        Name::new("Ground"),
        ComponentType::Ground,
    ));

    if let Some(meshes) = meshes.as_mut() {
        plane.insert(Mesh3d::from(
            meshes.add(Plane3d::new(Vec3::Y, Vec2::new(500.0, 500.0))),
        ));
    }
    if let Some(materials) = materials.as_mut() {
        plane.insert(MeshMaterial3d::from(
            materials.add(Color::srgb(1.0, 1.0, 1.0)),
        ));
    }

    // Spawn Cube Obstacle
    let mut cube = commands.spawn((
        Transform::from_xyz(250.0, 10.0, 250.0),
        Collider::cuboid(25.0, 25.0, 25.0),
        RigidBody::Fixed,
        CollisionGroups::new(Group::GROUP_2, Group::ALL),
        Friction::coefficient(0.8),
        ComponentType::Object,
    ));

    if let Some(meshes) = meshes.as_mut() {
        cube.insert(Mesh3d::from(meshes.add(Cuboid::new(50.0, 50.0, 50.0))));
    }
    if let Some(materials) = materials.as_mut() {
        cube.insert(MeshMaterial3d::from(
            materials.add(Color::srgb(0.5, 0.5, 0.1)),
        ));
    }

    // Spawn objective
    let mut objective = commands.spawn((
        Transform::from_xyz(-250.0, 10.0, -250.0),
        RigidBody::Fixed,
        Collider::cuboid(25.0, 25.0, 25.0),
        CollisionGroups::new(Group::GROUP_2, Group::ALL),
        Sensor,
        ActiveEvents::COLLISION_EVENTS,
        TriggerZone,
        ComponentType::Objective,
    ));

    if let Some(meshes) = meshes.as_mut() {
        objective.insert(Mesh3d::from(meshes.add(Cuboid::new(50.0, 50.0, 50.0))));
    }
    if let Some(materials) = materials.as_mut() {
        objective.insert(MeshMaterial3d::from(
            materials.add(Color::srgba(0.0, 1.0, 0.0, 0.3)),
        ));
    }

    if meshes.is_none() && materials.is_none() {
        return; // do not add lighting in headless mode
    }

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
