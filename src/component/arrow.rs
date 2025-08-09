use crate::component::player_character::PLAYER_HEIGHT;
use crate::player::Player;
use bevy::prelude::*;
use bevy::render::mesh::ConeAnchor;

// Marker — one per player
#[derive(Component)]
pub struct FacingArrow;

// Resource holding the mesh & material
#[derive(Resource)]
pub struct ArrowAssets {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
}

pub fn spawn_arrow_resource(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    mats: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let Some(mut meshes) = meshes else { return };
    let Some(mut mats) = mats else { return };
    let size = 20.0;
    // Shaft and head primitives
    let mut shaft = Cylinder::new(0.025 * size, 0.4 * size).mesh().build();
    let mut head = Cone::new(0.075 * size, 0.15 * size)
        .mesh()
        .anchor(ConeAnchor::Base)
        .build();

    // Move head tip to top of shaft
    head.translate_by(Vec3::Y * 0.2 * size);

    // Ensure that both have POSITION + NORMAL + UV compatible
    shaft.compute_normals();
    head.compute_normals();

    shaft.merge(&head).expect("incompatible attributes"); // merges both into one mesh :contentReference[oaicite:1]{index=1}

    let mesh = meshes.add(shaft);
    let material = mats.add(StandardMaterial {
        unlit: true,
        base_color: Color::srgb(1.0, 0.0, 0.0),
        ..Default::default()
    });

    commands.insert_resource(ArrowAssets { mesh, material })
}

pub fn spawn_arrows_to_players(
    mut commands: Commands,
    arrow: Option<Res<ArrowAssets>>,
    mut players: Query<(Entity, &Transform), With<Player>>,
) {
    let Some(arrow) = arrow else { return };

    for (player, transform) in players.iter_mut() {
        let forward = transform
            .rotation
            .mul_vec3(Vec3::Z)
            .try_normalize()
            .unwrap();
        commands.entity(player).with_children(|parent| {
            parent.spawn((
                FacingArrow,
                Transform {
                    translation: Vec3::Y * (PLAYER_HEIGHT * 0.5 + 2.0),
                    rotation: Quat::from_rotation_arc(Vec3::NEG_Y, forward),
                    scale: Vec3::ONE,
                },
                GlobalTransform::IDENTITY,
                Mesh3d(arrow.mesh.clone()),
                MeshMaterial3d(arrow.material.clone()),
            ));
        });
    }
}
