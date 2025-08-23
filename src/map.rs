use crate::ai::input::Input;
use crate::player::PLAYER_SPAWN_SAFE_DISTANCE;
use crate::sensor::objective::Objective;
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::pbr::{DirectionalLight, MeshMaterial3d, StandardMaterial};
use bevy::prelude::{default, Commands, Component, Mesh, Mesh3d, Name, Query, ResMut, Transform, Vec2, Vec3, With};
use bevy_math::prelude::{Cuboid, Plane3d};
use bevy_math::Quat;
use bevy_rapier3d::dynamics::RigidBody;
use bevy_rapier3d::geometry::{ActiveEvents, Collider, CollisionGroups, Friction, Group, Sensor};
use bevy_rapier3d::na::UnitQuaternion;
use rand::Rng;
use std::ops::{BitAnd, BitOr, BitOrAssign};

#[derive(Debug, Component, Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
pub enum ComponentType {
    None = 0,
    Ground = 1,
    Obstacle = 2,
    Objective = 3,
    Unknown = 4,
}

#[derive(Clone)]
pub struct MapObstacleSet {
    pub faces: Vec<ObstacleFaceSet>,
}

pub type ObstacleFaceSet = u8;

impl BitOrAssign<ObstacleFace> for u8 {
    fn bitor_assign(&mut self, rhs: ObstacleFace) {
        *self |= rhs as u8;
    }
}

impl BitOr for ObstacleFace {
    type Output = ObstacleFaceSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        self as u8 | rhs as u8
    }
}

impl BitAnd<ObstacleFace> for ObstacleFaceSet {
    type Output = ObstacleFaceSet;

    fn bitand(self, rhs: ObstacleFace) -> Self::Output {
        self as u8 & rhs as u8
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ObstacleFace {
    North = 1,
    South = 2,
    East = 4,
    West = 8,
}

pub const COMPONENTS_COUNT: usize = 31;
pub const MAP_SQUARE_SIZE: f32 = 500.0;
pub const COMPONENT_SIZE: f32 = 25.0;

pub type ComponentId = usize;

#[derive(Debug, Component)]
pub struct MapComponent(pub(crate) ComponentId);

#[derive(Debug, Component)]
pub struct RandomPositionComponent; // add this to a component to randomize its position each simulation reset

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

    let obstacle_mesh = meshes.as_deref_mut().map(|meshes| {
        Mesh3d::from(meshes.add(Cuboid::new(
            COMPONENT_SIZE * 2.0,
            COMPONENT_SIZE * 4.0,
            COMPONENT_SIZE * 2.0,
        )))
    });
    let obstacle_material = materials
        .as_deref_mut()
        .map(|materials| MeshMaterial3d::from(materials.add(Color::srgb(0.5, 0.5, 0.1))));

    // Spawn N-1 Cube Obstacles, and let one for the component
    for id in 0..(COMPONENTS_COUNT - 1) {
        let mut cube = commands.spawn((
            Transform::default(),
            Collider::cuboid(COMPONENT_SIZE, COMPONENT_SIZE * 2.0, COMPONENT_SIZE),
            RigidBody::Fixed,
            CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
            Friction::coefficient(0.8),
            ActiveEvents::COLLISION_EVENTS,
            ComponentType::Obstacle,
            MapComponent(id),
            RandomPositionComponent
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
        Transform::from_xyz(-MAP_SQUARE_SIZE, 0.0, -MAP_SQUARE_SIZE),
        RigidBody::Fixed,
        Collider::cuboid(COMPONENT_SIZE, COMPONENT_SIZE, COMPONENT_SIZE),
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        Sensor,
        ActiveEvents::COLLISION_EVENTS,
        Objective,
        ComponentType::Objective,
        MapComponent(COMPONENTS_COUNT - 1),
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

    spawn_walls(
        &mut commands,
        meshes.as_deref_mut(),
        materials.as_deref_mut(),
    );

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

pub fn spawn_walls(
    commands: &mut Commands,
    mut meshes: Option<&mut Assets<Mesh>>,
    materials: Option<&mut Assets<StandardMaterial>>,
) {
    // Dimensions
    let inner_height = 40.0;
    let tall_height = 4000.0;
    let thickness = 10.0;
    let collider_thickness = 100.0;
    let map_half = MAP_SQUARE_SIZE; // map goes from -MAP_SQUARE_SIZE .. +MAP_SQUARE_SIZE
    let map_length = MAP_SQUARE_SIZE * 2.0;
    let inner_y = inner_height * 0.5;
    let tall_y = tall_height * 0.5;
    // place the tall/outer walls slightly outside the visible inner walls
    let outer_offset = 25.0;

    // Meshes/material for visible inner walls (two mesh shapes: one long along X, one long along Z)
    let wall_mesh_x = meshes.as_deref_mut().map(|meshes| {
        Mesh3d::from(meshes.add(Cuboid::new(thickness, inner_height, map_length + 45.0)))
    });
    let wall_mesh_z = meshes.map(|meshes| {
        Mesh3d::from(meshes.add(Cuboid::new(map_length + 45.0, inner_height, thickness)))
    });
    let wall_material =
        materials.map(|materials| MeshMaterial3d::from(materials.add(Color::srgb(0.8, 0.8, 0.8))));

    // Helper spawn for an inner visible wall
    let spawn_inner_x = |commands: &mut Commands, x: f32| {
        let mut cmd = commands.spawn((
            Transform::from_xyz(x, inner_y, 0.0),
            ComponentType::Obstacle,
        ));
        if let Some(mesh) = wall_mesh_x.as_ref() {
            cmd.insert(mesh.clone());
        }
        if let Some(mat) = wall_material.as_ref() {
            cmd.insert(mat.clone());
        }
    };

    let spawn_inner_z = |commands: &mut Commands, z: f32| {
        let mut cmd = commands.spawn((
            Transform::from_xyz(0.0, inner_y, z),
            ComponentType::Obstacle,
        ));
        if let Some(mesh) = wall_mesh_z.as_ref() {
            cmd.insert(mesh.clone());
        }
        if let Some(mat) = wall_material.as_ref() {
            cmd.insert(mat.clone());
        }
    };

    // Inner visible walls (exactly around the map border)
    spawn_inner_x(commands, map_half + outer_offset); // +X
    spawn_inner_x(commands, -map_half - outer_offset); // -X
    spawn_inner_z(commands, map_half + outer_offset); // +Z
    spawn_inner_z(commands, -map_half - outer_offset); // -Z

    // +X outer
    commands.spawn((
        Transform::from_xyz(map_half + outer_offset + collider_thickness, tall_y, -50.0),
        Collider::cuboid(
            collider_thickness,
            tall_height * 0.5,
            map_half + outer_offset + collider_thickness * 2.0,
        ),
        ActiveEvents::COLLISION_EVENTS,
        RigidBody::Fixed,
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        ComponentType::Obstacle,
    ));
    // -X outer
    commands.spawn((
        Transform::from_xyz(-map_half - outer_offset - collider_thickness, tall_y, -50.0),
        Collider::cuboid(
            collider_thickness,
            tall_height * 0.5,
            map_half + outer_offset + collider_thickness * 2.0,
        ),
        ActiveEvents::COLLISION_EVENTS,
        RigidBody::Fixed,
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        ComponentType::Obstacle,
    ));
    // +Z outer
    commands.spawn((
        Transform::from_xyz(-50.0, tall_y, map_half + outer_offset + collider_thickness),
        Collider::cuboid(
            map_half + outer_offset + collider_thickness * 2.0,
            tall_height * 0.5,
            collider_thickness,
        ),
        ActiveEvents::COLLISION_EVENTS,
        RigidBody::Fixed,
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        ComponentType::Obstacle,
    ));
    // -Z outer
    commands.spawn((
        Transform::from_xyz(-50.0, tall_y, -map_half - outer_offset - collider_thickness),
        Collider::cuboid(
            map_half + outer_offset + collider_thickness * 2.0,
            tall_height * 0.5,
            collider_thickness,
        ),
        ActiveEvents::COLLISION_EVENTS,
        RigidBody::Fixed,
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        ComponentType::Obstacle,
    ));
}

pub fn reset_map(mut comps_q: Query<&mut Transform, With<RandomPositionComponent>>) {
    let mut rng = rand::rng();
    for mut comp in comps_q.iter_mut() {
        let spawn_area = MAP_SQUARE_SIZE - PLAYER_SPAWN_SAFE_DISTANCE;
        comp.translation = Vec3::new(
            rng.random_range(-spawn_area..spawn_area),
            COMPONENT_SIZE,
            rng.random_range(-spawn_area..spawn_area),
        );
        let yaw_deg = rng.random_range(0.0f32..360.0f32);
        let yaw = yaw_deg.to_radians();
        let unit_q = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
        comp.rotation = Quat::from(unit_q)
    }
}
