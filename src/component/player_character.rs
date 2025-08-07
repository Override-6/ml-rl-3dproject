use crate::human::player::RAYCASTS_DEBUG_DISPLAY_DISTANCE;
use crate::player::Player;
use crate::sensor::ground_sensor::GroundSensor;
use bevy::asset::Assets;
use bevy::color::palettes::basic::{BLUE, RED};
use bevy::color::Color;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    BevyError, Commands, Component, Entity, Gizmos, GlobalTransform, Mesh, Mesh3d, MeshRayCast,
    MeshRayCastSettings, Name, Query, ResMut, Transform, Vec3, With,
};
use bevy_math::prelude::Cuboid;
use bevy_math::{Dir3, Ray3d};
use bevy_rapier3d::dynamics::LockedAxes;
use bevy_rapier3d::geometry::{ActiveEvents, Collider, Sensor};

#[derive(Component)]
pub struct PlayerCharacter;

pub const PLAYER_WIDTH: f32 = 10.0;
pub const PLAYER_HEIGHT: f32 = 10.0;

pub fn spawn_player_character(
    players: Query<Entity, With<Player>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for player in players.iter() {
        commands.entity(player).with_children(|parent| {
            parent.spawn((
                Name::new("PlayerCharacter"),
                PlayerCharacter,
                Mesh3d(meshes.add(Cuboid::new(PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH))),
                MeshMaterial3d(materials.add(Color::srgb(0.7, 1.0, 0.5))),
                LockedAxes::ROTATION_LOCKED ^ LockedAxes::ROTATION_LOCKED_Y,
            ));
        });
        commands.entity(player).with_children(|parent| {
            // spawn ground detection
            parent.spawn((
                Name::new("GroundSensor"),
                GroundSensor,
                Collider::cuboid(PLAYER_WIDTH / 2.0, 0.1, PLAYER_WIDTH / 2.0),
                Sensor,
                ActiveEvents::COLLISION_EVENTS,
                Transform::from_xyz(0.0, -(PLAYER_HEIGHT / 2.0), 0.0),
                GlobalTransform::default(),
            ));
        });
    }
}

pub fn player_raycast_update(
    query: Query<(Entity, &GlobalTransform), With<PlayerCharacter>>,
    mut gizmos: Gizmos,
    mut mesh_ray_cast: MeshRayCast,
) -> bevy::prelude::Result<(), BevyError> {
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
