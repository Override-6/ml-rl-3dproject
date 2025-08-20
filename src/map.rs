use crate::sensor::objective::Objective;
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::pbr::{DirectionalLight, MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    default, Commands, Component, Mesh, Mesh3d, Name, Query, ResMut, Transform, Vec2, Vec3, With,
};
use bevy_math::prelude::{Cuboid, Plane3d};
use bevy_math::Quat;
use bevy_rapier3d::dynamics::RigidBody;
use bevy_rapier3d::geometry::{ActiveEvents, Collider, CollisionGroups, Friction, Group, Sensor};
use bevy_rapier3d::na::UnitQuaternion;
use rand::Rng;

#[derive(Debug, Component, Clone, Copy)]
#[repr(u8)]
pub enum ComponentType {
    None = 0,
    Ground = 1,
    Object = 2,
    Objective = 3,
    Unknown = 4,
}

pub const COMPONENTS_COUNT: usize = 20;
pub const MAP_SQUARE_SIZE: f32 = 500.0;
pub const COMPONENT_SIZE: f32 = 25.0;

#[derive(Debug, Component)]
pub struct MapComponent;

pub fn setup_map(
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    // Plane
    let mut plane = commands.spawn((
        Collider::halfspace(Vec3::Y).unwrap(),
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        Friction::coefficient(0.5),
        RigidBody::Fixed,
        Name::new("Ground"),
        ComponentType::Ground,
    ));

    if let Some(meshes) = meshes.as_mut() {
        plane.insert(Mesh3d::from(meshes.add(Plane3d::new(
            Vec3::Y,
            Vec2::new(MAP_SQUARE_SIZE + 25.0, MAP_SQUARE_SIZE + 25.0),
        ))));
    }
    if let Some(materials) = materials.as_mut() {
        plane.insert(MeshMaterial3d::from(
            materials.add(Color::srgb(1.0, 1.0, 1.0)),
        ));
    }

    let obstacle_mesh = meshes.as_deref_mut().map(|meshes| Mesh3d::from(meshes.add(Cuboid::new(
        COMPONENT_SIZE * 2.0,
        COMPONENT_SIZE * 2.0,
        COMPONENT_SIZE * 2.0,
    ))));
    let obstacle_material = materials.as_deref_mut().map(|materials| MeshMaterial3d::from(
        materials.add(Color::srgb(0.5, 0.5, 0.1)),
    ));

    // Spawn N Cube Obstacles
    for _ in 0..COMPONENTS_COUNT {
        let mut cube = commands.spawn((
            Transform::default(),
            Collider::cuboid(COMPONENT_SIZE, COMPONENT_SIZE, COMPONENT_SIZE),
            RigidBody::Fixed,
            CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
            Friction::coefficient(0.8),
            ComponentType::Object,
            MapComponent,
        ));

        if let Some(mesh) = obstacle_mesh.as_ref() {
            cube.insert(mesh.clone());
        }
        if let Some(material) = obstacle_material.as_ref() {
            cube.insert(material.clone());
        }
    }

    // Spawn objective
    let mut objective = commands.spawn((
        Transform::default(),
        RigidBody::Fixed,
        Collider::cuboid(COMPONENT_SIZE, COMPONENT_SIZE, COMPONENT_SIZE),
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        Sensor,
        ActiveEvents::COLLISION_EVENTS,
        Objective,
        ComponentType::Objective,
        MapComponent
    ));

    if let Some(meshes) = meshes.as_mut() {
        objective.insert(Mesh3d::from(meshes.add(Cuboid::new(
            COMPONENT_SIZE * 2.0,
            COMPONENT_SIZE * 2.0,
            COMPONENT_SIZE * 2.0,
        ))));
    }
    if let Some(materials) = materials.as_mut() {
        objective.insert(MeshMaterial3d::from(
            materials.add(Color::srgba(1.0, 0.2, 0.2, 0.4)),
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

pub fn reset_map(mut comps_q: Query<&mut Transform, With<MapComponent>>) {
    let mut rng = rand::rng();
    for mut comp in comps_q.iter_mut() {
        comp.translation = Vec3::new(
            rng.random_range(-MAP_SQUARE_SIZE..MAP_SQUARE_SIZE),
            COMPONENT_SIZE,
            rng.random_range(-MAP_SQUARE_SIZE..MAP_SQUARE_SIZE),
        );
        let yaw_deg = rng.random_range(0.0f32..360.0f32);
        let yaw = yaw_deg.to_radians();
        let unit_q = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
        comp.rotation = Quat::from(unit_q)
    }
}
