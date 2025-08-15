mod map;
mod player;
mod ui;
// unused
mod ai;
mod component;
mod drone;
mod human;
mod sensor;
mod simulation;

use crate::ai::input_recorder::GameInputRecorder;
use crate::ai::player::follow_all_script;
use crate::ai::script::Script;
use crate::component::arrow::{spawn_arrow_resource, spawn_arrows_to_players};
use crate::component::player_character::spawn_player_character;
use crate::human::camera_controller::{camera_follow, mouse_look, spawn_camera_controller};
use crate::human::player::move_player;
use crate::map::setup_map;
use crate::sensor::ground_sensor::ground_sensor_events;
use crate::sensor::player_vibrissae::{debug_render_lasers, update_all_vibrissae_lasers};
use crate::simulation::{DELTA_TIME, SimulationConfig, TICK_RATE, spawn_players};
use crate::ui::{setup_ui, update_stats_text};
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::pbr::DirectionalLightShadowMap;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use bevy_rapier3d::rapier::dynamics::IntegrationParameters;
use bincode::encode_into_slice;
use sensor::objective::{InTriggerZone, check_trigger_zone};
use std::cmp::PartialEq;
use std::fs::File;
use std::io::{Read, Write};
use crate::ai::model_control_pipeline::{setup_model_connection, sync_model_outputs, sync_state_outputs};

const NB_AI_PLAYERS: usize = 50;

#[derive(ScheduleLabel, Clone, Hash, PartialEq, Eq, Debug)]
pub struct PostPhysics;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum HeadMode {
    HeadRealTime,
    HeadRush,
    None,
}

fn main() {
    let script = read_script_from_file("script.bin");
    let app = create_app(HeadMode::HeadRealTime, script);
    run_simulation(app)
}

fn read_script_from_file(file: &str) -> Option<Script> {
    let file = File::open(file);
    let mut script = None;
    if let Ok(mut file) = file {
        let mut script_data = Vec::new();
        file.read_to_end(&mut script_data).unwrap();
        let decoded_script: Script =
            bincode::decode_from_slice(&script_data, bincode::config::standard())
                .unwrap()
                .0;
        script = Some(decoded_script);
    }

    script
}

fn create_app(head: HeadMode, script: Option<Script>) -> App {
    let mut app = App::new();

    // if true, rush as fast as possible
    let rush = head != HeadMode::HeadRealTime;

    if head == HeadMode::None {
        app.add_plugins(MinimalPlugins);
    } else {
        app.add_plugins(DefaultPlugins)
            .add_plugins(RapierDebugRenderPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(
                Startup,
                (setup_ui, spawn_arrow_resource, spawn_camera_controller),
            )
            .add_systems(
                Update,
                (
                    mouse_look,
                    camera_follow.after(PhysicsSet::Writeback),
                    update_stats_text,
                ),
            )
            .add_systems(
                Update,
                debug_render_lasers
                    .after(update_all_vibrissae_lasers)
                    .after(PhysicsSet::Writeback),
            )
            .add_systems(
                PostStartup,
                spawn_arrows_to_players.after(spawn_player_character),
            );
    }

    app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_systems(
            Startup,
            (
                setup,
                setup_map,
                spawn_players,
                spawn_player_character.after(spawn_players),
                setup_model_connection.after(spawn_player_character),
                sync_state_outputs.after(setup_model_connection)
            ),
        )
        .add_systems(Update, cleanup_on_exit)
        .insert_resource(
            RapierContextInitialization::InitializeDefaultRapierContext {
                integration_parameters: IntegrationParameters {
                    dt: DELTA_TIME,
                    ..default()
                },
                rapier_configuration: RapierConfiguration {
                    ..RapierConfiguration::new(100f32)
                },
            },
        )
        .insert_resource(DirectionalLightShadowMap { size: 100 });

    if rush {
        app.insert_resource(TimestepMode::Fixed {
            dt: DELTA_TIME,
            substeps: 1,
        });
    } else {
        // run relative to a fixed tick rate
        app.insert_resource(Time::<Fixed>::from_hz(TICK_RATE as f64));
        app.insert_resource(TimestepMode::Variable {
            max_dt: DELTA_TIME,
            time_scale: 1.0,
            substeps: 1,
        });
    };

    macro_rules! add_game_logic_systems {
        ($systems: expr) => {
            if rush {
                app.add_systems(Update, $systems)
            } else {
                app.add_systems(FixedUpdate, $systems)
            }
        };
    }

    if let Some(script) = script {
        app.insert_resource(SimulationConfig::Simulation {
            script,
            num_ai_players: NB_AI_PLAYERS,
        });
        add_game_logic_systems!(follow_all_script);
    } else {
        assert_eq!(
            head,
            HeadMode::HeadRealTime,
            "Must be in HeadMode::RealTime if you want to record inputs"
        );
        app.insert_resource(GameInputRecorder::new());
        app.insert_resource(SimulationConfig::Game);
        add_game_logic_systems!(move_player);
    }

    add_game_logic_systems!((
        ground_sensor_events,
        check_trigger_zone,
        update_all_vibrissae_lasers,
        sync_model_outputs.before(follow_all_script),
        sync_state_outputs.after(PhysicsSet::StepSimulation),
    ));

    app
}

fn run_simulation(mut app: App) {
    app.run();
}

fn setup(mut commands: Commands) {
    commands.insert_resource(InTriggerZone(false));
}

fn cleanup_on_exit(
    mut events: EventReader<AppExit>,
    recorded_input: Option<Res<GameInputRecorder>>,
) -> Result<(), BevyError> {
    if let Some(event) = events.read().next() {
        info!("App is exiting with status: {:?}", event);

        let Some(recorded_input) = recorded_input else {
            return Ok(());
        };

        let recorded_input: Script = recorded_input.as_ref().into();

        let mut file = File::create("script.bin")?;
        let mut buff = [0; 10000];
        encode_into_slice(&recorded_input, &mut buff, bincode::config::standard())?;
        file.write_all(&buff)?;
    }
    Ok(())
}
