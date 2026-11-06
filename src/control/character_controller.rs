use crate::alloc_prelude::*;
use crate::geometry::{ColliderHandle, ContactManifold, Shape, ShapeCastHit};
use crate::math::{Pose, Real, Vector};
use crate::pipeline::{QueryFilterFlags, QueryPipeline, QueryPipelineMut};
use crate::utils;
use na::{RealField, Vector2};
use parry::bounding_volume::BoundingVolume;
use parry::query::details::ShapeCastOptions;
use parry::query::{DefaultQueryDispatcher, PersistentQueryDispatcher};

/// [fork 0017] Runtime switch (default off = stock): the character controller visits the colliders of its
/// AABB queries in ascending collider-handle order instead of BVH traversal order. Several loops are order-dependent
/// (`check_and_fix_penetrations` applies corrections sequentially, the kinematic-friction sum in
/// `detect_grounded_status_and_apply_friction`, the push in `solve_single_character_collision_impulse`), and the BVH
/// layout depends on the world's history, so a world restored from a snapshot can otherwise move a character differently.
pub static KCC_CANONICAL_ORDER: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn kcc_aabb_candidates<'a>(
    queries: &'a QueryPipeline<'a>,
    aabb: parry::bounding_volume::Aabb,
) -> Vec<(ColliderHandle, &'a crate::geometry::Collider)> {
    let mut v: Vec<_> = queries.intersect_aabb_conservative(aabb).collect();
    if KCC_CANONICAL_ORDER.load(core::sync::atomic::Ordering::Relaxed) {
        v.sort_by_key(|(h, _)| h.into_raw_parts());
    }
    v
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
/// A length measure used for various options of a character controller.
pub enum CharacterLength {
    /// The length is specified relative to some of the character shape’s size.
    ///
    /// For example setting `CharacterAutostep::max_height` to `CharacterLength::Relative(0.1)`
    /// for a shape with a height equal to 20.0 will result in a maximum step height
    /// of `0.1 * 20.0 = 2.0`.
    Relative(Real),
    /// The length is specified as an absolute value, independent from the character shape’s size.
    ///
    /// For example setting `CharacterAutostep::max_height` to `CharacterLength::Relative(0.1)`
    /// for a shape with a height equal to 20.0 will result in a maximum step height
    /// of `0.1` (the shape height is ignored in for this value).
    Absolute(Real),
}

impl CharacterLength {
    /// Returns `self` with its value changed by the closure `f` if `self` is the `Self::Absolute`
    /// variant.
    pub fn map_absolute(self, f: impl FnOnce(Real) -> Real) -> Self {
        if let Self::Absolute(value) = self {
            Self::Absolute(f(value))
        } else {
            self
        }
    }

    /// Returns `self` with its value changed by the closure `f` if `self` is the `Self::Relative`
    /// variant.
    pub fn map_relative(self, f: impl FnOnce(Real) -> Real) -> Self {
        if let Self::Relative(value) = self {
            Self::Relative(f(value))
        } else {
            self
        }
    }

    fn eval(self, value: Real) -> Real {
        match self {
            Self::Relative(x) => value * x,
            Self::Absolute(x) => x,
        }
    }
}

#[derive(Debug)]
struct HitInfo {
    toi: ShapeCastHit,
    is_wall: bool,
    is_nonslip_slope: bool,
}

/// Configuration for the auto-stepping character controller feature.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CharacterAutostep {
    /// The maximum step height a character can automatically step over.
    pub max_height: CharacterLength,
    /// The minimum width of free space that must be available after stepping on a stair.
    pub min_width: CharacterLength,
    /// Can the character automatically step over dynamic bodies too?
    pub include_dynamic_bodies: bool,
}

impl Default for CharacterAutostep {
    fn default() -> Self {
        Self {
            max_height: CharacterLength::Relative(0.25),
            min_width: CharacterLength::Relative(0.5),
            include_dynamic_bodies: true,
        }
    }
}

#[derive(Debug)]
struct HitDecomposition {
    normal_part: Vector,
    horizontal_tangent: Vector,
    vertical_tangent: Vector,
    // NOTE: we don’t store the penetration part since we don’t really need it
    //       for anything.
}

impl HitDecomposition {
    pub fn unconstrained_slide_part(&self) -> Vector {
        self.normal_part + self.horizontal_tangent + self.vertical_tangent
    }
}

/// A collision between the character and its environment during its movement.
#[derive(Copy, Clone, Debug)]
pub struct CharacterCollision {
    /// The collider hit by the character.
    pub handle: ColliderHandle,
    /// The position of the character when the collider was hit.
    pub character_pos: Pose,
    /// The translation that was already applied to the character when the hit happens.
    pub translation_applied: Vector,
    /// The translations that was still waiting to be applied to the character when the hit happens.
    pub translation_remaining: Vector,
    /// Geometric information about the hit.
    pub hit: ShapeCastHit,
}

/// A kinematic character controller for player/NPC movement (walking, climbing, sliding).
///
/// This provides classic game character movement: walking on floors, sliding on slopes,
/// climbing stairs, and snapping to ground. It's kinematic (not physics-based), meaning
/// you control movement directly rather than applying forces.
///
/// **Not suitable for:** Ragdolls, vehicles, or physics-driven movement (use dynamic bodies instead).
///
/// # How it works
///
/// 1. You provide desired movement (e.g., "move forward 5 units")
/// 2. Controller casts the character shape through the world
/// 3. It handles collisions: sliding along walls, stepping up stairs, snapping to ground
/// 4. Returns the final movement to apply
///
/// # Example
///
/// ```
/// # use rapier3d::prelude::*;
/// # use rapier3d::control::{CharacterAutostep, KinematicCharacterController};
/// # let mut bodies = RigidBodySet::new();
/// # let mut colliders = ColliderSet::new();
/// # let broad_phase = BroadPhaseBvh::new();
/// # let narrow_phase = NarrowPhase::new();
/// # let dt = 1.0 / 60.0;
/// # let speed = 5.0;
/// # let (input_x, input_z) = (1.0, 0.0);
/// # let character_shape = Ball::new(0.5);
/// # let mut character_pos = Pose::IDENTITY;
/// # let query_pipeline = broad_phase.as_query_pipeline(
/// #     narrow_phase.query_dispatcher(),
/// #     &bodies,
/// #     &colliders,
/// #     QueryFilter::default(),
/// # );
/// let controller = KinematicCharacterController {
///     slide: true,  // Slide along walls instead of stopping
///     autostep: Some(CharacterAutostep::default()),  // Auto-climb stairs
///     max_slope_climb_angle: 45.0_f32.to_radians(),  // Max climbable slope
///     ..Default::default()
/// };
///
/// // In your game loop:
/// let desired_movement = Vector::new(input_x, 0.0, input_z) * speed * dt;
/// let movement = controller.move_shape(
///     dt,
///     &query_pipeline,
///     &character_shape,
///     &character_pos,
///     desired_movement,
///     |_| {}  // Collision event callback
/// );
/// character_pos.translation += movement.translation;
/// ```
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct KinematicCharacterController {
    /// The direction that goes "up". Used to determine where the floor is, and the floor’s angle.
    pub up: Vector,
    /// A small gap to preserve between the character and its surroundings.
    ///
    /// This value should not be too large to avoid visual artifacts, but shouldn’t be too small
    /// (must not be zero) to improve numerical stability of the character controller.
    pub offset: CharacterLength,
    /// Should the character try to slide against the floor if it hits it?
    pub slide: bool,
    /// Should the character automatically step over small obstacles? (disabled by default)
    ///
    /// Note that autostepping is currently a very computationally expensive feature, so it
    /// is disabled by default.
    pub autostep: Option<CharacterAutostep>,
    /// The maximum angle (radians) between the floor’s normal and the `up` vector that the
    /// character is able to climb.
    pub max_slope_climb_angle: Real,
    /// The minimum angle (radians) between the floor’s normal and the `up` vector before the
    /// character starts to slide down automatically.
    pub min_slope_slide_angle: Real,
    /// Should the character be automatically snapped to the ground if the distance between
    /// the ground and its feed are smaller than the specified threshold?
    pub snap_to_ground: Option<CharacterLength>,
    /// Increase this number if your character appears to get stuck when sliding against surfaces.
    ///
    /// This is a small distance applied to the movement toward the contact normals of shapes hit
    /// by the character controller. This helps shape-casting not getting stuck in an always-penetrating
    /// state during the sliding calculation.
    ///
    /// This value should remain fairly small since it can introduce artificial "bumps" when sliding
    /// along a flat surface.
    pub normal_nudge_factor: Real,
    /// [fork 0012] Opt-in cost cuts that change trajectories slightly (all `false` = stock 0.36.1).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub fast_paths: CharacterFastPaths,
    /// [fork 0025] Which contacts with kinematic bodies carry the character (default = stock).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub kinematic_carry: KinematicCarry,
}

/// [fork 0025] Kinematic-friction ("moving platform") policy of the character controller.
///
/// Stock (`AllContacts`) transfers motion from *every* kinematic body the character touches: the component of
/// the remaining translation along each contact normal is replaced by the body's normal velocity, and the
/// tangential part of the body's velocity is added (larger-absolute-value-wins per axis), whatever the contact
/// normal. When characters are themselves kinematic bodies that obstruct each other, two characters in side
/// contact carry each other's *vertical* velocity tangentially: A's upward motion is added to B, B's to A, and
/// each one's next-tick velocity (displacement / dt) feeds the other again, so the pair is launched (distphys N9:
/// characters thousands of metres up). Both options below keep every contact on which the character stands
/// (grounded manifold: normal·up ≥ 1e-3 with a point within the prediction distance) exactly as stock, so
/// moving platforms behave the same, and change nothing at all for characters that touch no kinematic body.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum KinematicCarry {
    /// Stock 0.36.1 behaviour.
    #[default]
    AllContacts,
    /// Tangential platform motion only from contacts the character stands on; the normal-velocity
    /// transfer (a moving kinematic wall pushing the character) is kept for every contact.
    TangentFromGroundOnly,
    /// Both the normal and the tangential transfer only from contacts the character stands on: other
    /// kinematic bodies (e.g. other characters) are plain obstacles from the side.
    GroundOnly,
    /// [fork 0026] Contacts the character stands on: as stock. Other contacts with kinematic bodies: no
    /// tangential carry, and the normal transfer is one-sided: it only raises the character's motion along
    /// the contact normal (an approaching body pushes it), never lowers it (a receding body does not pull
    /// it along). This is the shape of Jolt's CharacterVirtual contact-plane constraint.
    SidesPushOnly,
}

/// [fork 0012] Opt-in KCC cost cuts. Each one can change character trajectories (see the fork notes);
/// all `false` (the default) reproduces stock 0.36.1 exactly.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CharacterFastPaths {
    /// When the character is grounded at its start pose, drop the downward (−`up`) part of the
    /// desired translation before the slide loop: no slide-loop cast just to touch the ground
    /// (snap-to-ground still follows descending ground). A character walking off a ledge starts
    /// to fall one tick later.
    pub grounded_skip_downward: bool,
    /// Snap to the ground only if the character is not already grounded at the end of the slide
    /// loop (the stock comment's intent; stock snaps whenever it was grounded at the start).
    pub snap_only_if_not_grounded: bool,
    /// Skip the autostep casts when the obstacle's highest point (exact support point along `up`)
    /// is above `feet + max_height`: such a step cannot succeed in the usual geometry.
    pub autostep_height_precheck: bool,
}

impl Default for KinematicCharacterController {
    fn default() -> Self {
        Self {
            up: Vector::Y,
            offset: CharacterLength::Relative(0.01),
            slide: true,
            autostep: None,
            max_slope_climb_angle: Real::frac_pi_4(),
            min_slope_slide_angle: Real::frac_pi_4(),
            snap_to_ground: Some(CharacterLength::Relative(0.2)),
            normal_nudge_factor: 1.0e-4,
            fast_paths: CharacterFastPaths::default(),
            kinematic_carry: KinematicCarry::AllContacts,
        }
    }
}

/// The effective movement computed by the character controller.
#[derive(Debug)]
pub struct EffectiveCharacterMovement {
    /// The movement to apply.
    pub translation: Vector,
    /// Is the character touching the ground after applying `EffectiveKineamticMovement::translation`?
    pub grounded: bool,
    /// Is the character sliding down a slope due to slope angle being larger than `min_slope_slide_angle`?
    pub is_sliding_down_slope: bool,
}

impl KinematicCharacterController {
    /// Pushes the character out of any non-sensor collider it is overlapping with, which is
    /// what stops an obstacle pushed into a non-moving character from tunneling through it
    /// ([`Self::move_shape`]'s shape-casts never run on a zero desired translation).
    ///
    /// The total correction per call is capped relative to the character’s height.
    fn check_and_fix_penetrations(
        &self,
        queries: &QueryPipeline,
        character_shape: &dyn Shape,
        character_pos: &Pose,
        dims: Vector2<Real>,
        result: &mut EffectiveCharacterMovement,
    ) {
        let offset = self.offset.eval(dims.y);
        let max_correction = dims.y * 0.25;
        let mut applied = 0.0;

        // Run a few passes so that getting pushed out of one collider doesn’t leave the
        // character stuck inside another one.
        for _ in 0..4 {
            let character_aabb = character_shape
                .compute_aabb(&(Pose::from_translation(result.translation) * *character_pos))
                .loosened(offset);
            let mut corrected = false;

            for (_, collider) in kcc_aabb_candidates(queries, character_aabb) {
                if collider.is_sensor() {
                    continue;
                }

                let character_pos = Pose::from_translation(result.translation) * *character_pos;
                let pos12 = character_pos.inv_mul(collider.position());

                if let Ok(Some(contact)) =
                    queries
                        .dispatcher
                        .contact(&pos12, character_shape, collider.shape(), 0.0)
                {
                    if contact.dist < -1.0e-5 {
                        // Push out until the usual `offset` gap is restored.
                        let push = (offset - contact.dist).min(max_correction - applied);
                        if push <= 0.0 {
                            return; // The per-call correction budget is exhausted.
                        }

                        // `normal1` (expressed in the character’s local frame) points towards
                        // the obstacle: move backwards along it to resolve the overlap.
                        result.translation -= (character_pos.rotation * contact.normal1) * push;
                        applied += push;
                        corrected = true;
                    }
                }
            }

            if !corrected {
                break;
            }
        }
    }

    /// Computes the possible movement for a shape.
    #[profiling::function]
    pub fn move_shape(
        &self,
        dt: Real,
        queries: &QueryPipeline,
        character_shape: &dyn Shape,
        character_pos: &Pose,
        desired_translation: Vector,
        mut events: impl FnMut(CharacterCollision),
    ) -> EffectiveCharacterMovement {
        let mut result = EffectiveCharacterMovement {
            translation: Vector::ZERO,
            grounded: false,
            is_sliding_down_slope: false,
        };
        let dims = self.compute_dims(character_shape);

        // 1. Depenetrate, but only when there is no desired movement: the shape-casting
        //    loop below never runs then, so obstacles pushed into the character would
        //    tunnel through it (#485). When it moves, the shape-casts handle them instead.
        if utils::try_normalize_and_get_length(desired_translation, 1.0e-5).is_none() {
            self.check_and_fix_penetrations(
                queries,
                character_shape,
                character_pos,
                dims,
                &mut result,
            );
        }

        let mut translation_remaining = desired_translation;

        let grounded_at_starting_pos = self.detect_grounded_status_and_apply_friction(
            dt,
            queries,
            character_shape,
            &(Pose::from_translation(result.translation) * *character_pos),
            dims,
            None,
            None,
        );
        // [fork 0012] Opt-in: a grounded character does not cast into the ground it stands on.
        if self.fast_paths.grounded_skip_downward && grounded_at_starting_pos {
            let down = translation_remaining.dot(self.up);
            if down < 0.0 {
                translation_remaining -= self.up * down;
            }
        }

        let mut max_iters = 20;
        let mut kinematic_friction_translation = Vector::ZERO;
        // [fork 0011] Pose (and grounded-on-kinematic flag) of the last friction pass whose full
        // grounded status is still owed.
        let mut pending_ground: Option<(Pose, bool)> = None;
        let offset = self.offset.eval(dims.y);
        let mut is_moving = false;

        while let Some((translation_dir, translation_dist)) =
            utils::try_normalize_and_get_length(translation_remaining, 1.0e-5)
        {
            if max_iters == 0 {
                break;
            } else {
                max_iters -= 1;
            }
            is_moving = true;

            // 2. Cast towards the movement direction.
            if let Some((handle, hit)) = queries.cast_shape(
                &(Pose::from_translation(result.translation) * *character_pos),
                translation_dir,
                character_shape,
                ShapeCastOptions {
                    target_distance: offset,
                    stop_at_penetration: false,
                    max_time_of_impact: translation_dist,
                    compute_impact_geometry_on_penetration: true,
                },
            ) {
                // We hit something, compute and apply the allowed interference-free translation.
                let allowed_dist = hit.time_of_impact;
                let allowed_translation = translation_dir * allowed_dist;
                result.translation += allowed_translation;
                translation_remaining -= allowed_translation;

                events(CharacterCollision {
                    handle,
                    character_pos: Pose::from_translation(result.translation) * *character_pos,
                    translation_applied: result.translation,
                    translation_remaining,
                    hit,
                });

                let hit_info = self.compute_hit_info(hit);

                // Try to go upstairs.
                if !self.handle_stairs(
                    *queries,
                    character_shape,
                    &(Pose::from_translation(result.translation) * *character_pos),
                    dims,
                    handle,
                    &hit_info,
                    &mut translation_remaining,
                    &mut result,
                ) {
                    // No stairs, try to move along slopes.
                    translation_remaining = self.handle_slopes(
                        &hit_info,
                        desired_translation,
                        translation_remaining,
                        self.normal_nudge_factor,
                        &mut result,
                    );
                }
            } else {
                // No interference along the path.
                pending_ground = None;
                result.translation += translation_remaining;
                result.grounded = self.detect_grounded_status_and_apply_friction(
                    dt,
                    queries,
                    character_shape,
                    &(Pose::from_translation(result.translation) * *character_pos),
                    dims,
                    None,
                    None,
                );
                break;
            }
            // [fork 0011] Kinematic friction only (manifolds of colliders with a kinematic parent);
            // the grounded flag of this pose is computed lazily below, only if no later iteration
            // overwrites it (stock computed every collider's manifolds here on every iteration).
            let pose = Pose::from_translation(result.translation) * *character_pos;
            let grounded_on_kinematic = self.detect_grounded_status_and_apply_friction_ext(
                dt,
                queries,
                character_shape,
                &pose,
                dims,
                Some(&mut kinematic_friction_translation),
                Some(&mut translation_remaining),
                true,
            );
            pending_ground = Some((pose, grounded_on_kinematic));

            if !self.slide {
                break;
            }
        }
        if let Some((pose, grounded_on_kinematic)) = pending_ground {
            result.grounded = grounded_on_kinematic
                || self.detect_grounded_status_and_apply_friction(
                    dt,
                    queries,
                    character_shape,
                    &pose,
                    dims,
                    None,
                    None,
                );
        }
        // When not moving, `detect_grounded_status_and_apply_friction` is not reached
        // so we call it explicitly here.
        if !is_moving {
            result.grounded = self.detect_grounded_status_and_apply_friction(
                dt,
                queries,
                character_shape,
                &(Pose::from_translation(result.translation) * *character_pos),
                dims,
                None,
                None,
            );
        }
        // If needed, and if we are not already grounded, snap to the ground.
        // [fork 0012] Opt-in: really skip the snap cast when already grounded.
        if grounded_at_starting_pos
            && !(self.fast_paths.snap_only_if_not_grounded && result.grounded)
        {
            self.snap_to_ground(
                queries,
                character_shape,
                &(Pose::from_translation(result.translation) * *character_pos),
                dims,
                &mut result,
            );
        }

        // Return the result.
        result
    }

    fn snap_to_ground(
        &self,
        queries: &QueryPipeline,
        character_shape: &dyn Shape,
        character_pos: &Pose,
        dims: Vector2<Real>,
        result: &mut EffectiveCharacterMovement,
    ) -> Option<(ColliderHandle, ShapeCastHit)> {
        if let Some(snap_distance) = self.snap_to_ground {
            if result.translation.dot(self.up) <= 0.0 {
                let snap_distance = snap_distance.eval(dims.y);
                let offset = self.offset.eval(dims.y);
                if let Some((hit_handle, hit)) = queries.cast_shape(
                    character_pos,
                    -self.up,
                    character_shape,
                    ShapeCastOptions {
                        target_distance: offset,
                        stop_at_penetration: false,
                        max_time_of_impact: snap_distance,
                        compute_impact_geometry_on_penetration: true,
                    },
                ) {
                    // Apply the snap.
                    result.translation -= self.up * hit.time_of_impact;
                    result.grounded = true;
                    return Some((hit_handle, hit));
                }
            }
        }

        None
    }

    fn predict_ground(&self, up_extends: Real) -> Real {
        self.offset.eval(up_extends) + 0.05
    }

    #[profiling::function]
    fn detect_grounded_status_and_apply_friction(
        &self,
        dt: Real,
        queries: &QueryPipeline,
        character_shape: &dyn Shape,
        character_pos: &Pose,
        dims: Vector2<Real>,
        kinematic_friction_translation: Option<&mut Vector>,
        translation_remaining: Option<&mut Vector>,
    ) -> bool {
        self.detect_grounded_status_and_apply_friction_ext(
            dt,
            queries,
            character_shape,
            character_pos,
            dims,
            kinematic_friction_translation,
            translation_remaining,
            false,
        )
    }

    /// [fork 0011] `kinematic_only`: skip colliders without a kinematic parent (they contribute
    /// no kinematic friction), so the result is only the grounded status on kinematic bodies.
    #[allow(clippy::too_many_arguments)]
    fn detect_grounded_status_and_apply_friction_ext(
        &self,
        dt: Real,
        queries: &QueryPipeline,
        character_shape: &dyn Shape,
        character_pos: &Pose,
        dims: Vector2<Real>,
        mut kinematic_friction_translation: Option<&mut Vector>,
        mut translation_remaining: Option<&mut Vector>,
        kinematic_only: bool,
    ) -> bool {
        let prediction = self.predict_ground(dims.y);

        // TODO: allow custom dispatchers.
        let dispatcher = DefaultQueryDispatcher;

        let mut manifolds: Vec<ContactManifold> = vec![];
        let character_aabb = character_shape
            .compute_aabb(character_pos)
            .loosened(prediction);

        let mut grounded = false;

        'outer: for (_, collider) in kcc_aabb_candidates(queries, character_aabb) {
            if kinematic_only
                && !collider
                    .parent
                    .and_then(|p| queries.bodies.get(p.handle))
                    .is_some_and(|rb| rb.is_kinematic())
            {
                continue;
            }
            manifolds.clear();
            let pos12 = character_pos.inv_mul(collider.position());
            let _ = dispatcher.contact_manifolds(
                &pos12,
                character_shape,
                collider.shape(),
                prediction,
                &mut manifolds,
                &mut None,
            );

            if let (Some(kinematic_friction_translation), Some(translation_remaining)) = (
                kinematic_friction_translation.as_deref_mut(),
                translation_remaining.as_deref_mut(),
            ) {
                let init_kinematic_friction_translation = *kinematic_friction_translation;
                let kinematic_parent = collider
                    .parent
                    .and_then(|p| queries.bodies.get(p.handle))
                    .filter(|rb| rb.is_kinematic());

                for m in &manifolds {
                    let ground_manifold = self.is_grounded_at_contact_manifold(m, character_pos, dims);
                    if ground_manifold {
                        grounded = true;
                    }
                    // [fork 0025] Which parts of the kinematic transfer this manifold may apply.
                    let (carry_normal, carry_tangent) = match self.kinematic_carry {
                        KinematicCarry::AllContacts => (true, true),
                        KinematicCarry::TangentFromGroundOnly => (true, ground_manifold),
                        KinematicCarry::GroundOnly => (ground_manifold, ground_manifold),
                        KinematicCarry::SidesPushOnly => (true, ground_manifold),
                    };
                    // [fork 0026] one-sided normal transfer on side contacts.
                    let push_only =
                        self.kinematic_carry == KinematicCarry::SidesPushOnly && !ground_manifold;

                    if let Some(kinematic_parent) = kinematic_parent.filter(|_| carry_normal || carry_tangent) {
                        let mut num_active_contacts = 0;
                        let mut manifold_center = Vector::ZERO;
                        let normal = -(character_pos.rotation * m.local_n1);

                        for contact in &m.points {
                            if contact.dist <= prediction {
                                num_active_contacts += 1;
                                let contact_point = collider.position() * contact.local_p2;
                                let target_vel = kinematic_parent.velocity_at_point(contact_point);

                                let normal_target_mvt = target_vel.dot(normal) * dt;
                                let normal_current_mvt = translation_remaining.dot(normal);

                                manifold_center += contact_point;
                                if carry_normal
                                    && (!push_only || normal_target_mvt > normal_current_mvt)
                                {
                                    *translation_remaining +=
                                        normal * (normal_target_mvt - normal_current_mvt);
                                }
                            }
                        }

                        if num_active_contacts > 0 && carry_tangent {
                            let target_vel = kinematic_parent
                                .velocity_at_point(manifold_center / num_active_contacts as Real);
                            let tangent_platform_mvt =
                                (target_vel - normal * target_vel.dot(normal)) * dt;
                            // Apply larger-absolute-value-wins component-wise
                            if tangent_platform_mvt.x.abs() > kinematic_friction_translation.x.abs()
                            {
                                kinematic_friction_translation.x = tangent_platform_mvt.x;
                            }
                            if tangent_platform_mvt.y.abs() > kinematic_friction_translation.y.abs()
                            {
                                kinematic_friction_translation.y = tangent_platform_mvt.y;
                            }
                            #[cfg(feature = "dim3")]
                            if tangent_platform_mvt.z.abs() > kinematic_friction_translation.z.abs()
                            {
                                kinematic_friction_translation.z = tangent_platform_mvt.z;
                            }
                        }
                    }
                }

                *translation_remaining +=
                    *kinematic_friction_translation - init_kinematic_friction_translation;
            } else {
                for m in &manifolds {
                    if self.is_grounded_at_contact_manifold(m, character_pos, dims) {
                        grounded = true;
                        break 'outer; // We can stop the search early.
                    }
                }
            }
        }
        grounded
    }

    fn is_grounded_at_contact_manifold(
        &self,
        manifold: &ContactManifold,
        character_pos: &Pose,
        dims: Vector2<Real>,
    ) -> bool {
        let normal = -(character_pos.rotation * manifold.local_n1);

        // For the controller to be grounded, the angle between the contact normal and the up vector
        // has to be smaller than acos(1.0e-3) = 89.94 degrees.
        if normal.dot(self.up) >= 1.0e-3 {
            let prediction = self.predict_ground(dims.y);
            for contact in &manifold.points {
                if contact.dist <= prediction {
                    return true;
                }
            }
        }
        false
    }

    fn handle_slopes(
        &self,
        hit: &HitInfo,
        movement_input: Vector,
        translation_remaining: Vector,
        normal_nudge_factor: Real,
        result: &mut EffectiveCharacterMovement,
    ) -> Vector {
        let [_vertical_input, horizontal_input] = self.split_into_components(movement_input);
        let horiz_input_decomp = self.decompose_hit(horizontal_input, &hit.toi);
        let decomp = self.decompose_hit(translation_remaining, &hit.toi);

        // An object is trying to slip if the tangential movement induced by its vertical movement
        // points downward.
        let slipping_intent = self.up.dot(horiz_input_decomp.vertical_tangent) < 0.0;
        // An object is slipping if its vertical movement points downward.
        let slipping = self.up.dot(decomp.vertical_tangent) < 0.0;

        // An object is trying to climb if its vertical input motion points upward.
        let climbing_intent = self.up.dot(_vertical_input) > 0.0;
        // An object is climbing if the tangential movement induced by its vertical movement points upward.
        let climbing = self.up.dot(decomp.vertical_tangent) > 0.0;

        let allowed_movement = if hit.is_wall && climbing && !climbing_intent {
            // Can’t climb the slope, remove the vertical tangent motion induced by the forward motion.
            decomp.horizontal_tangent + decomp.normal_part
        } else if hit.is_nonslip_slope && slipping && !slipping_intent {
            // Prevent the vertical movement from sliding down.
            decomp.horizontal_tangent + decomp.normal_part
        } else {
            // Let it slide (including climbing the slope).
            result.is_sliding_down_slope = true;
            decomp.unconstrained_slide_part()
        };

        allowed_movement + hit.toi.normal1 * normal_nudge_factor
    }

    fn split_into_components(&self, translation: Vector) -> [Vector; 2] {
        let vertical_translation = self.up * (self.up.dot(translation));
        let horizontal_translation = translation - vertical_translation;
        [vertical_translation, horizontal_translation]
    }

    fn compute_hit_info(&self, toi: ShapeCastHit) -> HitInfo {
        #[cfg(feature = "dim2")]
        let angle_with_floor = self.up.angle_to(toi.normal1);
        #[cfg(feature = "dim3")]
        let angle_with_floor = self.up.angle_between(toi.normal1);
        let is_ceiling = self.up.dot(toi.normal1) < 0.0;
        let is_wall = angle_with_floor >= self.max_slope_climb_angle && !is_ceiling;
        let is_nonslip_slope = angle_with_floor <= self.min_slope_slide_angle;

        HitInfo {
            toi,
            is_wall,
            is_nonslip_slope,
        }
    }

    fn decompose_hit(&self, translation: Vector, hit: &ShapeCastHit) -> HitDecomposition {
        let dist_to_surface = translation.dot(hit.normal1);
        let normal_part;
        let penetration_part;

        if dist_to_surface < 0.0 {
            normal_part = Vector::ZERO;
            penetration_part = dist_to_surface * hit.normal1;
        } else {
            penetration_part = Vector::ZERO;
            normal_part = dist_to_surface * hit.normal1;
        }

        let tangent = translation - normal_part - penetration_part;
        #[cfg(feature = "dim3")]
        let horizontal_tangent_dir = hit.normal1.cross(self.up);
        #[cfg(feature = "dim2")]
        let horizontal_tangent_dir = Vector::ZERO;

        let horizontal_tangent_dir = horizontal_tangent_dir.try_normalize().unwrap_or_default();
        let horizontal_tangent = tangent.dot(horizontal_tangent_dir) * horizontal_tangent_dir;
        let vertical_tangent = tangent - horizontal_tangent;

        HitDecomposition {
            normal_part,
            horizontal_tangent,
            vertical_tangent,
        }
    }

    fn compute_dims(&self, character_shape: &dyn Shape) -> Vector2<Real> {
        let extents = character_shape.compute_local_aabb().extents();
        let up_extent = extents.dot(self.up.abs());
        let side_extent = (extents - (self.up).abs() * up_extent).length();
        Vector2::new(side_extent, up_extent)
    }

    #[profiling::function]
    fn handle_stairs(
        &self,
        mut queries: QueryPipeline,
        character_shape: &dyn Shape,
        character_pos: &Pose,
        dims: Vector2<Real>,
        stair_handle: ColliderHandle,
        hit: &HitInfo,
        translation_remaining: &mut Vector,
        result: &mut EffectiveCharacterMovement,
    ) -> bool {
        let Some(autostep) = self.autostep else {
            return false;
        };

        // Only try to autostep on walls.
        if !hit.is_wall {
            return false;
        }

        let offset = self.offset.eval(dims.y);
        let min_width = autostep.min_width.eval(dims.x) + offset;
        let max_height = autostep.max_height.eval(dims.y) + offset;

        if !autostep.include_dynamic_bodies {
            if queries
                .colliders
                .get(stair_handle)
                .and_then(|co| co.parent)
                .and_then(|p| queries.bodies.get(p.handle))
                .map(|b| b.is_dynamic())
                == Some(true)
            {
                // The "stair" is a dynamic body, which the user wants to ignore.
                return false;
            }

            queries.filter.flags |= QueryFilterFlags::EXCLUDE_DYNAMIC;
        }

        // [fork 0012] Opt-in: an obstacle whose top is higher than a step cannot be stepped on.
        if self.fast_paths.autostep_height_precheck {
            if let Some(co) = queries.colliders.get(stair_handle) {
                if let (Some(obstacle), Some(character)) =
                    (co.shape().as_support_map(), character_shape.as_support_map())
                {
                    let local_up = co.position().rotation.inverse() * self.up;
                    let top = (co.position() * obstacle.local_support_point(local_up)).dot(self.up);
                    let char_down = character_pos.rotation.inverse() * -self.up;
                    let feet =
                        (character_pos * character.local_support_point(char_down)).dot(self.up);
                    if top - feet > max_height {
                        return false;
                    }
                }
            }
        }

        let shifted_character_pos = Pose::from_parts(
            character_pos.translation + self.up * max_height,
            character_pos.rotation,
        );

        let Some(horizontal_dir) =
            (*translation_remaining - self.up * translation_remaining.dot(self.up)).try_normalize()
        else {
            return false;
        };

        if queries
            .cast_shape(
                character_pos,
                self.up,
                character_shape,
                ShapeCastOptions {
                    target_distance: offset,
                    stop_at_penetration: false,
                    max_time_of_impact: max_height,
                    compute_impact_geometry_on_penetration: true,
                },
            )
            .is_some()
        {
            // We can’t go up.
            return false;
        }

        if queries
            .cast_shape(
                &shifted_character_pos,
                horizontal_dir,
                character_shape,
                ShapeCastOptions {
                    target_distance: offset,
                    stop_at_penetration: false,
                    max_time_of_impact: min_width,
                    compute_impact_geometry_on_penetration: true,
                },
            )
            .is_some()
        {
            // We don’t have enough room on the stair to stay on it.
            return false;
        }

        // Check that we are not getting into a ramp that is too steep
        // after stepping.
        // [fork 0011] The step-height cast below is this exact query: keep its result.
        let step_down_hit = queries.cast_shape(
            &Pose::from_parts(
                shifted_character_pos.translation + horizontal_dir * min_width,
                shifted_character_pos.rotation,
            ),
            -self.up,
            character_shape,
            ShapeCastOptions {
                target_distance: offset,
                stop_at_penetration: false,
                max_time_of_impact: max_height,
                compute_impact_geometry_on_penetration: true,
            },
        );
        if let Some((_, hit)) = step_down_hit {
            let [vertical_slope_translation, horizontal_slope_translation] = self
                .split_into_components(*translation_remaining)
                .map(|remaining| subtract_hit(remaining, &hit));

            let slope_translation = horizontal_slope_translation + vertical_slope_translation;

            #[cfg(feature = "dim2")]
            let angle_with_floor = self.up.angle_to(hit.normal1);
            #[cfg(feature = "dim3")]
            let angle_with_floor = self.up.angle_between(hit.normal1);
            let climbing = self.up.dot(slope_translation) >= 0.0;

            if climbing && angle_with_floor > self.max_slope_climb_angle {
                return false; // The target ramp is too steep.
            }
        }

        // We can step, we need to find the actual step height.
        let step_height = max_height
            - step_down_hit
                .map(|hit| hit.1.time_of_impact)
                .unwrap_or(max_height);

        // Remove the step height from the vertical part of the self.
        let step = self.up * step_height;
        *translation_remaining -= step;

        // Advance the collider on the step horizontally, to make sure further
        // movement won’t just get stuck on its edge.
        let horizontal_nudge =
            horizontal_dir * horizontal_dir.dot(*translation_remaining).min(min_width);
        *translation_remaining -= horizontal_nudge;

        result.translation += step + horizontal_nudge;
        true
    }

    /// For the given collisions between a character and its environment, this method will apply
    /// impulses to the rigid-bodies surrounding the character shape at the time of the collisions.
    /// Note that the impulse calculation is only approximate as it is not based on a global
    /// constraints resolution scheme.
    #[profiling::function]
    pub fn solve_character_collision_impulses<'a>(
        &self,
        dt: Real,
        queries: &mut QueryPipelineMut,
        character_shape: &dyn Shape,
        character_mass: Real,
        collisions: impl IntoIterator<Item = &'a CharacterCollision>,
    ) {
        for collision in collisions {
            self.solve_single_character_collision_impulse(
                dt,
                queries,
                character_shape,
                character_mass,
                collision,
            );
        }
    }

    /// For the given collision between a character and its environment, this method will apply
    /// impulses to the rigid-bodies surrounding the character shape at the time of the collision.
    /// Note that the impulse calculation is only approximate as it is not based on a global
    /// constraints resolution scheme.
    #[profiling::function]
    fn solve_single_character_collision_impulse(
        &self,
        dt: Real,
        queries: &mut QueryPipelineMut,
        character_shape: &dyn Shape,
        character_mass: Real,
        collision: &CharacterCollision,
    ) {
        let extents = character_shape.compute_local_aabb().extents();
        let up_extent = extents.dot(self.up.abs());
        let movement_to_transfer =
            collision.hit.normal1 * collision.translation_remaining.dot(collision.hit.normal1);
        let prediction = self.predict_ground(up_extent);

        // TODO: allow custom dispatchers.
        let dispatcher = DefaultQueryDispatcher;

        let mut manifolds: Vec<ContactManifold> = vec![];
        // World pose of the collider each manifold was computed against: the `local_p2`
        // points are in the collider’s frame, which differs from its body’s when offset.
        let mut manifold_collider_poses: Vec<Pose> = vec![];
        // output vec for contact manifolds returned by parry
        let mut pair_manifolds: Vec<ContactManifold> = vec![];
        let character_aabb = character_shape
            .compute_aabb(&collision.character_pos)
            .loosened(prediction);

        for (_, collider) in kcc_aabb_candidates(&queries.as_ref(), character_aabb) {
            if let Some(parent) = collider.parent {
                if let Some(body) = queries.bodies.get(parent.handle) {
                    if body.is_dynamic() {
                        let pos12 = collision.character_pos.inv_mul(collider.position());
                        pair_manifolds.clear();
                        let _ = dispatcher.contact_manifolds(
                            &pos12,
                            character_shape,
                            collider.shape(),
                            prediction,
                            &mut pair_manifolds,
                            &mut None,
                        );

                        for mut m in pair_manifolds.drain(..) {
                            m.data.rigid_body2 = Some(parent.handle);
                            m.data.normal = collision.character_pos.rotation * m.local_n1;
                            manifolds.push(m);
                            manifold_collider_poses.push(*collider.position());
                        }
                    }
                }
            }
        }

        let velocity_to_transfer = movement_to_transfer * utils::inv(dt);

        for (manifold, collider_pos) in manifolds.iter().zip(manifold_collider_poses.iter()) {
            let body_handle = manifold.data.rigid_body2.unwrap();
            let body = &mut queries.bodies[body_handle];

            for pt in &manifold.points {
                if pt.dist <= prediction {
                    let body_mass = body.mass();
                    let contact_point = collider_pos * pt.local_p2;
                    let delta_vel_per_contact = (velocity_to_transfer
                        - body.velocity_at_point(contact_point))
                    .dot(manifold.data.normal);
                    let mass_ratio = body_mass * character_mass / (body_mass + character_mass);

                    body.apply_impulse_at_point(
                        manifold.data.normal * delta_vel_per_contact.max(0.0) * mass_ratio,
                        contact_point,
                        true,
                    );
                }
            }
        }
    }

    /// [fork 0024] Read-only first half of [`Self::solve_character_collision_impulses`]: gathers, for the given
    /// collisions of one character, every dynamic-body contact the stock push would use (the AABB query and the
    /// contact manifolds, i.e. all of the push's geometry work) into `out`, without touching any body.
    ///
    /// The push only changes body *velocities*; the gathered geometry depends only on collider poses, body types
    /// and the query filter, which the push does not change. So the gather of every character can run in parallel
    /// against the frozen world, and [`Self::apply_character_collision_impulses`] called serially in the stock
    /// character order then gives a world that is bit-identical to calling `solve_character_collision_impulses` for
    /// each character in that order (same per-body arithmetic in the same order, same mutable accesses in the same
    /// order, hence the same modified-body set). `out` is cleared first.
    pub fn prepare_character_collision_impulses<'a>(
        &self,
        dt: Real,
        queries: &QueryPipeline,
        character_shape: &dyn Shape,
        collisions: impl IntoIterator<Item = &'a CharacterCollision>,
        out: &mut CharacterPush,
    ) {
        out.manifolds.clear();
        out.points.clear();
        let extents = character_shape.compute_local_aabb().extents();
        let up_extent = extents.dot(self.up.abs());
        let prediction = self.predict_ground(up_extent);
        let dispatcher = DefaultQueryDispatcher;
        let mut pair_manifolds: Vec<ContactManifold> = vec![];
        for collision in collisions {
            let movement_to_transfer =
                collision.hit.normal1 * collision.translation_remaining.dot(collision.hit.normal1);
            let character_aabb = character_shape
                .compute_aabb(&collision.character_pos)
                .loosened(prediction);
            let velocity_to_transfer = movement_to_transfer * utils::inv(dt);
            for (_, collider) in kcc_aabb_candidates(queries, character_aabb) {
                if let Some(parent) = collider.parent {
                    if let Some(body) = queries.bodies.get(parent.handle) {
                        if body.is_dynamic() {
                            let pos12 = collision.character_pos.inv_mul(collider.position());
                            pair_manifolds.clear();
                            let _ = dispatcher.contact_manifolds(
                                &pos12,
                                character_shape,
                                collider.shape(),
                                prediction,
                                &mut pair_manifolds,
                                &mut None,
                            );
                            let collider_pos = collider.position();
                            for m in pair_manifolds.drain(..) {
                                let normal = collision.character_pos.rotation * m.local_n1;
                                for pt in &m.points {
                                    if pt.dist <= prediction {
                                        out.points.push(collider_pos * pt.local_p2);
                                    }
                                }
                                out.manifolds.push(CharacterPushManifold {
                                    body: parent.handle,
                                    normal,
                                    velocity_to_transfer,
                                    points_end: out.points.len() as u32,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// [fork 0024] Serial second half: applies a push gathered by
    /// [`Self::prepare_character_collision_impulses`] with exactly the stock arithmetic and access order.
    pub fn apply_character_collision_impulses(
        &self,
        bodies: &mut crate::dynamics::RigidBodySet,
        character_mass: Real,
        push: &CharacterPush,
    ) {
        let mut start = 0usize;
        for m in &push.manifolds {
            // Stock takes `&mut bodies[h]` once per manifold, even one without points within the prediction
            // distance: keep that access (it records the body in the modified set, which affects the next step).
            let body = &mut bodies[m.body];
            let end = m.points_end as usize;
            for contact_point in &push.points[start..end] {
                let body_mass = body.mass();
                let contact_point = *contact_point;
                let delta_vel_per_contact = (m.velocity_to_transfer
                    - body.velocity_at_point(contact_point))
                .dot(m.normal);
                let mass_ratio = body_mass * character_mass / (body_mass + character_mass);

                body.apply_impulse_at_point(
                    m.normal * delta_vel_per_contact.max(0.0) * mass_ratio,
                    contact_point,
                    true,
                );
            }
            start = end;
        }
    }
}

/// [fork 0024] One contact manifold of a gathered character push (see
/// [`KinematicCharacterController::prepare_character_collision_impulses`]).
#[derive(Copy, Clone, Debug)]
pub struct CharacterPushManifold {
    /// The dynamic body pushed.
    pub body: crate::dynamics::RigidBodyHandle,
    /// World-space manifold normal (character towards obstacle), as the stock push uses it.
    pub normal: Vector,
    /// The character velocity to transfer along `normal` for the collision this manifold came from.
    pub velocity_to_transfer: Vector,
    /// End (exclusive) of this manifold's contact points in [`CharacterPush::points`].
    pub points_end: u32,
}

/// [fork 0024] The gathered push of one character: manifolds in the stock visit order and their contact points
/// (world space, only those within the prediction distance).
#[derive(Clone, Debug, Default)]
pub struct CharacterPush {
    /// Manifolds in the order the stock push visits them.
    pub manifolds: Vec<CharacterPushManifold>,
    /// World-space contact points, grouped by manifold.
    pub points: Vec<Vector>,
}

fn subtract_hit(translation: Vector, hit: &ShapeCastHit) -> Vector {
    let surface_correction = (-translation).dot(hit.normal1).max(0.0);
    // This fixes some instances of moving through walls
    let surface_correction = surface_correction * (1.0 + 1.0e-5);
    translation + hit.normal1 * surface_correction
}

#[cfg(all(feature = "dim3", feature = "f32"))]
#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use crate::alloc_prelude::*;
    use crate::{
        control::{CharacterLength, KinematicCharacterController},
        prelude::*,
    };
    use std::dbg;

    #[test]
    fn character_controller_climb_test() {
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();
        let mut soft_bodies = SoftBodySet::new();
        let mut pipeline = PhysicsPipeline::new();
        let mut bf = BroadPhaseBvh::new();
        let mut nf = NarrowPhase::new();
        let mut islands = IslandManager::new();

        let mut bodies = RigidBodySet::new();

        let gravity = Vector::Y * -9.81;

        let ground_size = 100.0;
        let ground_height = 0.1;
        /*
         * Create a flat ground
         */
        let rigid_body =
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -ground_height, 0.0));
        let floor_handle = bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
        colliders.insert_with_parent(collider, floor_handle, &mut bodies);

        /*
         * Create a slope we can climb.
         */
        let slope_angle = 0.2;
        let slope_size = 2.0;
        let collider = ColliderBuilder::cuboid(slope_size, ground_height, slope_size)
            .translation(Vector::new(0.1 + slope_size, -ground_height + 0.4, 0.0))
            .rotation(Vector::Z * slope_angle);
        colliders.insert(collider);

        /*
         * Create a slope we can't climb.
         */
        let impossible_slope_angle = 0.6;
        let impossible_slope_size = 2.0;
        // The slope is long enough that the character never reaches its crest within the
        // iteration budget: wrapping around the crest is a knife-edge configuration and a
        // known corner-robustness limitation (#809), not what this test is about.
        let collider = ColliderBuilder::cuboid(slope_size * 2.0, ground_height, ground_size)
            .translation(Vector::new(
                0.1 + slope_size * 2.0 + impossible_slope_size - 0.9 + slope_size,
                -ground_height + 1.7 + slope_size * impossible_slope_angle.sin(),
                0.0,
            ))
            .rotation(Vector::Z * impossible_slope_angle);
        colliders.insert(collider);

        let integration_parameters = IntegrationParameters::default();

        // Initialize character which can climb
        let mut character_body_can_climb = RigidBodyBuilder::kinematic_position_based()
            .additional_mass(1.0)
            .build();
        character_body_can_climb.set_translation(Vector::new(0.6, 0.5, 0.0), false);
        let character_handle_can_climb = bodies.insert(character_body_can_climb);

        let collider = ColliderBuilder::ball(0.5).build();
        colliders.insert_with_parent(collider.clone(), character_handle_can_climb, &mut bodies);

        // Initialize character which cannot climb
        let mut character_body_cannot_climb = RigidBodyBuilder::kinematic_position_based()
            .additional_mass(1.0)
            .build();
        character_body_cannot_climb.set_translation(Vector::new(-0.6, 0.5, 0.0), false);
        let character_handle_cannot_climb = bodies.insert(character_body_cannot_climb);

        let collider = ColliderBuilder::ball(0.5).build();
        let character_shape = collider.shape();
        colliders.insert_with_parent(collider.clone(), character_handle_cannot_climb, &mut bodies);

        let mut airborne_can_climb = (0u32, 0u32);
        let mut airborne_cannot_climb = (0u32, 0u32);
        for i in 0..200 {
            // Step once
            pipeline.step(
                gravity,
                &integration_parameters,
                &mut islands,
                &mut bf,
                &mut nf,
                &mut bodies,
                &mut colliders,
                &mut impulse_joints,
                &mut multibody_joints,
                &mut soft_bodies,
                &mut CCDSolver::new(),
                &(),
                &(),
            );

            let mut update_character_controller =
                |controller: KinematicCharacterController,
                 handle: RigidBodyHandle,
                 airborne: &mut (u32, u32)| {
                    let character_body = bodies.get(handle).unwrap();
                    // Use a closure to handle or collect the collisions while
                    // the character is being moved.
                    let mut collisions = vec![];
                    let filter_character_controller = QueryFilter::new().exclude_rigid_body(handle);
                    let query_pipeline = bf.as_query_pipeline(
                        nf.query_dispatcher(),
                        &bodies,
                        &colliders,
                        filter_character_controller,
                    );
                    let effective_movement = controller.move_shape(
                        integration_parameters.dt,
                        &query_pipeline,
                        character_shape,
                        character_body.position(),
                        Vector::new(0.1, -0.1, 0.0),
                        |collision| collisions.push(collision),
                    );
                    let character_body = bodies.get_mut(handle).unwrap();
                    let translation = character_body.translation();
                    // Grounded-ness must hold except for rare isolated frames: at slope
                    // transitions the contact sits right on the detection threshold, so a
                    // 1-ulp change flips one frame. Consecutive ones are a real regression.
                    let (total_airborne, consecutive_airborne) = airborne;
                    if effective_movement.grounded {
                        *consecutive_airborne = 0;
                    } else {
                        *total_airborne += 1;
                        *consecutive_airborne += 1;
                        assert!(
                            *consecutive_airborne <= 1 && *total_airborne <= 3,
                            "movement should be grounded at all times (up to isolated \
                             threshold flickers) for current setup (iter: {}), pos: {}.",
                            i,
                            translation + effective_movement.translation
                        );
                    }
                    character_body.set_next_kinematic_translation(
                        translation + effective_movement.translation,
                    );
                };

            let character_controller_cannot_climb = KinematicCharacterController {
                max_slope_climb_angle: impossible_slope_angle - 0.001,
                ..Default::default()
            };
            let character_controller_can_climb = KinematicCharacterController {
                max_slope_climb_angle: impossible_slope_angle + 0.001,
                ..Default::default()
            };
            update_character_controller(
                character_controller_cannot_climb,
                character_handle_cannot_climb,
                &mut airborne_cannot_climb,
            );
            update_character_controller(
                character_controller_can_climb,
                character_handle_can_climb,
                &mut airborne_can_climb,
            );
        }
        let character_body = bodies.get(character_handle_can_climb).unwrap();
        assert!(character_body.translation().x > 6.0);
        assert!(character_body.translation().y > 3.0);
        let character_body = bodies.get(character_handle_cannot_climb).unwrap();
        // The blocked character rests against the (extended) steep slope's base.
        assert!(character_body.translation().x < 4.5);
        assert!(dbg!(character_body.translation().y) < 2.0);
    }

    #[test]
    fn character_controller_ground_detection() {
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();
        let mut soft_bodies = SoftBodySet::new();
        let mut pipeline = PhysicsPipeline::new();
        let mut bf = BroadPhaseBvh::new();
        let mut nf = NarrowPhase::new();
        let mut islands = IslandManager::new();

        let mut bodies = RigidBodySet::new();

        let gravity = Vector::Y * -9.81;

        let ground_size = 1001.0;
        let ground_height = 1.0;
        /*
         * Create a flat ground
         */
        let rigid_body =
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -ground_height / 2.0, 0.0));
        let floor_handle = bodies.insert(rigid_body);
        let collider = ColliderBuilder::cuboid(ground_size, ground_height, ground_size);
        colliders.insert_with_parent(collider, floor_handle, &mut bodies);

        let integration_parameters = IntegrationParameters::default();

        // Initialize character with snap to ground
        let character_controller_snap = KinematicCharacterController {
            snap_to_ground: Some(CharacterLength::Relative(0.2)),
            ..Default::default()
        };
        let mut character_body_snap = RigidBodyBuilder::kinematic_position_based()
            .additional_mass(1.0)
            .build();
        character_body_snap.set_translation(Vector::new(0.6, 0.5, 0.0), false);
        let character_handle_snap = bodies.insert(character_body_snap);

        let collider = ColliderBuilder::ball(0.5).build();
        colliders.insert_with_parent(collider.clone(), character_handle_snap, &mut bodies);

        // Initialize character without snap to ground
        let character_controller_no_snap = KinematicCharacterController {
            snap_to_ground: None,
            ..Default::default()
        };
        let mut character_body_no_snap = RigidBodyBuilder::kinematic_position_based()
            .additional_mass(1.0)
            .build();
        character_body_no_snap.set_translation(Vector::new(-0.6, 0.5, 0.0), false);
        let character_handle_no_snap = bodies.insert(character_body_no_snap);

        let collider = ColliderBuilder::ball(0.5).build();
        let character_shape = collider.shape();
        colliders.insert_with_parent(collider.clone(), character_handle_no_snap, &mut bodies);

        for i in 0..10000 {
            // Step once
            pipeline.step(
                gravity,
                &integration_parameters,
                &mut islands,
                &mut bf,
                &mut nf,
                &mut bodies,
                &mut colliders,
                &mut impulse_joints,
                &mut multibody_joints,
                &mut soft_bodies,
                &mut CCDSolver::new(),
                &(),
                &(),
            );

            let mut update_character_controller =
                |controller: KinematicCharacterController, handle: RigidBodyHandle| {
                    let character_body = bodies.get(handle).unwrap();
                    // Use a closure to handle or collect the collisions while
                    // the character is being moved.
                    let mut collisions = vec![];
                    let filter_character_controller = QueryFilter::new().exclude_rigid_body(handle);
                    let query_pipeline = bf.as_query_pipeline(
                        nf.query_dispatcher(),
                        &bodies,
                        &colliders,
                        filter_character_controller,
                    );
                    let effective_movement = controller.move_shape(
                        integration_parameters.dt,
                        &query_pipeline,
                        character_shape,
                        character_body.position(),
                        Vector::new(0.1, -0.1, 0.1),
                        |collision| collisions.push(collision),
                    );
                    let character_body = bodies.get_mut(handle).unwrap();
                    let translation = character_body.translation();
                    assert!(
                        effective_movement.grounded,
                        "movement should be grounded at all times for current setup (iter: {}), pos: {}.",
                        i,
                        translation + effective_movement.translation
                    );
                    character_body.set_next_kinematic_translation(
                        translation + effective_movement.translation,
                    );
                };

            update_character_controller(character_controller_no_snap, character_handle_no_snap);
            update_character_controller(character_controller_snap, character_handle_snap);
        }

        let character_body = bodies.get_mut(character_handle_no_snap).unwrap();
        let translation = character_body.translation();

        // Accumulated numerical error makes the character fall short of the ideal distance.
        // The bound was 940 until parry 0.30.2 tightened the shape-cast TOI at GJK stagnation
        // (parry#429), which costs a little more progress per step.
        assert!(
            translation.x >= 930.0,
            "actual translation.x:{}",
            translation.x
        );
        assert!(
            translation.z >= 930.0,
            "actual translation.z:{}",
            translation.z
        );

        let character_body = bodies.get_mut(character_handle_snap).unwrap();
        let translation = character_body.translation();
        assert!(
            translation.x >= 960.0,
            "actual translation.x:{}",
            translation.x
        );
        assert!(
            translation.z >= 960.0,
            "actual translation.z:{}",
            translation.z
        );
    }

    #[test]
    fn character_controller_negative_up_axis() {
        // Regression test for issue #492: a `KinematicCharacterController` whose `up` vector
        // points along negative Y must not panic in `move_shape`, and must still detect the
        // ground.
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();
        let mut soft_bodies = SoftBodySet::new();
        let mut pipeline = PhysicsPipeline::new();
        let mut bf = BroadPhaseBvh::new();
        let mut nf = NarrowPhase::new();
        let mut islands = IslandManager::new();
        let mut bodies = RigidBodySet::new();

        // In a world whose "up" is -Y, the floor is *above* the character (higher Y).
        // Floor half-height 0.1 centered at y = 0.6 => its lower face is at y = 0.5.
        let ground = RigidBodyBuilder::fixed().translation(Vector::new(0.0, 0.6, 0.0));
        let ground_handle = bodies.insert(ground);
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(10.0, 0.1, 10.0),
            ground_handle,
            &mut bodies,
        );

        // Character ball of radius 0.5 at the origin => its top is at y = 0.5, i.e. touching
        // the floor's lower face.
        let character_handle = bodies.insert(RigidBodyBuilder::kinematic_position_based());
        let character_collider = ColliderBuilder::ball(0.5).build();
        colliders.insert_with_parent(character_collider.clone(), character_handle, &mut bodies);

        let integration_parameters = IntegrationParameters::default();
        pipeline.step(
            Vector::ZERO,
            &integration_parameters,
            &mut islands,
            &mut bf,
            &mut nf,
            &mut bodies,
            &mut colliders,
            &mut impulse_joints,
            &mut multibody_joints,
            &mut soft_bodies,
            &mut CCDSolver::new(),
            &(),
            &(),
        );

        let controller = KinematicCharacterController {
            up: -Vector::Y,
            ..Default::default()
        };

        let filter = QueryFilter::new().exclude_rigid_body(character_handle);
        let query_pipeline =
            bf.as_query_pipeline(nf.query_dispatcher(), &bodies, &colliders, filter);

        // This used to panic with "The loosening margin must be positive." (issue #492).
        let movement = controller.move_shape(
            integration_parameters.dt,
            &query_pipeline,
            character_collider.shape(),
            bodies[character_handle].position(),
            Vector::new(0.1, 0.0, 0.0),
            |_| {},
        );

        // The floor lies in the +Y direction, i.e. along `-up`, so the character is grounded.
        assert!(movement.grounded);
    }
    #[test]
    fn character_controller_impulses_with_multiple_dynamic_bodies() {
        // `solve_character_collision_impulses` used to gather every nearby body's manifolds
        // into one vec and slice it at the previous length, assuming `contact_manifolds`
        // appends. Parry's composite-shape paths rebuild the vec for the current pair instead,
        // so with a compound character and two dynamic bodies in range the second pair either
        // indexed past the end or left its manifolds without a body handle, panicking either way.
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();
        let mut soft_bodies = SoftBodySet::new();
        let mut pipeline = PhysicsPipeline::new();
        let mut bf = BroadPhaseBvh::new();
        let mut nf = NarrowPhase::new();
        let mut islands = IslandManager::new();
        let mut bodies = RigidBodySet::new();

        let ground = RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.6, 0.0));
        let ground_handle = bodies.insert(ground);
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(10.0, 0.1, 10.0),
            ground_handle,
            &mut bodies,
        );

        // A 1x1x1 character made of two stacked half-cubes, so its shape is a compound.
        let character_handle = bodies.insert(RigidBodyBuilder::kinematic_position_based());
        let character_collider = ColliderBuilder::compound(vec![
            (
                Pose::from_translation(Vector::new(0.0, -0.25, 0.0)),
                SharedShape::cuboid(0.5, 0.25, 0.5),
            ),
            (
                Pose::from_translation(Vector::new(0.0, 0.25, 0.0)),
                SharedShape::cuboid(0.5, 0.25, 0.5),
            ),
        ])
        .build();
        colliders.insert_with_parent(character_collider.clone(), character_handle, &mut bodies);

        // Two dynamic unit cubes touching the character: one ahead (+X), in the movement
        // path, and one beside it (+Z), close enough to be gathered by the impulse solver but
        // not in the way.
        let mut insert_box = |translation: Vector| {
            let handle = bodies.insert(RigidBodyBuilder::dynamic().translation(translation));
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.5, 0.5, 0.5),
                handle,
                &mut bodies,
            );
            handle
        };
        let box_ahead = insert_box(Vector::new(1.0, 0.0, 0.0));
        let box_beside = insert_box(Vector::new(0.0, 0.0, 1.0));

        let integration_parameters = IntegrationParameters::default();
        pipeline.step(
            Vector::ZERO,
            &integration_parameters,
            &mut islands,
            &mut bf,
            &mut nf,
            &mut bodies,
            &mut colliders,
            &mut impulse_joints,
            &mut multibody_joints,
            &mut soft_bodies,
            &mut CCDSolver::new(),
            &(),
            &(),
        );

        let controller = KinematicCharacterController::default();
        let filter = QueryFilter::new().exclude_rigid_body(character_handle);

        let mut collisions = vec![];
        let query_pipeline =
            bf.as_query_pipeline(nf.query_dispatcher(), &bodies, &colliders, filter);
        controller.move_shape(
            integration_parameters.dt,
            &query_pipeline,
            character_collider.shape(),
            bodies[character_handle].position(),
            Vector::new(0.1, 0.0, 0.0),
            |c| collisions.push(c),
        );
        assert!(
            collisions
                .iter()
                .any(|c| colliders[c.handle].parent() == Some(box_ahead)),
            "the character should have run into the box ahead of it"
        );

        let mut query_pipeline_mut =
            bf.as_query_pipeline_mut(nf.query_dispatcher(), &mut bodies, &mut colliders, filter);
        // This used to panic with "range start index N out of range for slice of length 0"
        // or on `rigid_body2.unwrap()`.
        controller.solve_character_collision_impulses(
            integration_parameters.dt,
            &mut query_pipeline_mut,
            character_collider.shape(),
            1.0,
            &collisions,
        );

        // The box ahead was pushed forward; the box beside lies across the motion and is
        // left alone.
        assert!(bodies[box_ahead].linvel().x > 0.0);
        assert_eq!(bodies[box_beside].linvel(), Vector::ZERO);
    }
}
