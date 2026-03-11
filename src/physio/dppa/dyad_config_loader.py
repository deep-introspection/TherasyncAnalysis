"""
Dyad Configuration Loader for DPPA Analysis.

This module loads and parses the dyad configuration file (dppa_dyads_real.yaml)
which defines real therapeutic dyads and their metadata for inter-session
and intra-session comparisons.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Set
from itertools import combinations

logger = logging.getLogger(__name__)


class DyadConfigLoader:
    """
    Load and parse dyad configuration from YAML file.

    This class provides methods to retrieve dyad pairs for both:
    - Intra-session: Real dyads within same session (sliding window epoching)
    - Inter-session: All possible pairs for baseline comparison (nsplit epoching)

    Attributes:
        config_path: Path to dyad configuration YAML file
        config: Parsed configuration dictionary
        _real_dyads_set: Set of (subject1, subject2, session) for O(1) lookup

    Example:
        >>> loader = DyadConfigLoader()
        >>> real_dyads = loader.get_real_dyads(family='g01', session='ses-01')
        >>> for dyad in real_dyads:
        ...     print(f"{dyad['subject1']} <-> {dyad['subject2']}: {dyad['dyad_type']}")
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DyadConfigLoader.

        Args:
            config_path: Path to dyad config file. If None, uses default.
        """
        if config_path is None:
            config_path = Path("config/dppa_dyads_real.yaml")
        else:
            config_path = Path(config_path)

        self.config_path = config_path

        # Load configuration
        if not config_path.exists():
            raise FileNotFoundError(f"Dyad config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Validate configuration structure
        self._validate_config()

        # Build lookup set for real dyads (for O(1) is_real_dyad checks)
        self._real_dyads_set = self._build_real_dyads_set()

        logger.info(f"Dyad Config Loader initialized from {config_path}")
        logger.info(f"  {len(self.config.get('real_dyads', []))} real dyads loaded")

    def _validate_config(self):
        """Validate configuration structure and required keys."""
        # Check for main sections
        if "epoching" not in self.config:
            raise ValueError("Missing 'epoching' section in config")
        if "families" not in self.config:
            raise ValueError("Missing 'families' section in config")
        if "real_dyads" not in self.config:
            raise ValueError("Missing 'real_dyads' section in config")

        # Validate epoching
        epoching = self.config["epoching"]
        if "intra_session" not in epoching:
            raise ValueError("Missing 'intra_session' in epoching config")
        if "inter_session" not in epoching:
            raise ValueError("Missing 'inter_session' in epoching config")

        # Validate intra_session (requires single 'method')
        if "method" not in epoching["intra_session"]:
            raise ValueError("Missing 'method' in epoching.intra_session")
        if "tasks" not in epoching["intra_session"]:
            raise ValueError("Missing 'tasks' in epoching.intra_session")

        # Validate inter_session (accepts either 'method' or 'methods')
        inter_config = epoching["inter_session"]
        if "method" not in inter_config and "methods" not in inter_config:
            raise ValueError("Missing 'method' or 'methods' in epoching.inter_session")
        if "tasks" not in inter_config:
            raise ValueError("Missing 'tasks' in epoching.inter_session")

        logger.debug("Configuration validation passed")

    def _build_real_dyads_set(self) -> Set[Tuple[str, str, str]]:
        """Build a set of (subject1, subject2, session) for fast lookup."""
        real_set = set()
        for dyad in self.config.get("real_dyads", []):
            s1 = dyad["subject1"].replace("sub-", "")
            s2 = dyad["subject2"].replace("sub-", "")
            ses = dyad["session"]
            # Store both orderings for lookup
            real_set.add((s1, s2, ses))
            real_set.add((s2, s1, ses))
        return real_set

    # =========================================================================
    # Real Dyads Methods
    # =========================================================================

    def get_real_dyads(
        self,
        family: Optional[str] = None,
        session: Optional[str] = None,
        dyad_type: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get real therapeutic dyads with optional filters.

        Args:
            family: Filter by family ID (e.g., 'g01')
            session: Filter by session (e.g., 'ses-01')
            dyad_type: Filter by type ('therapist-patient' or 'patient-patient')
            task: Validate task is in config (e.g., 'therapy')

        Returns:
            List of dyad dictionaries with keys:
            - dyad_id, family, subject1, role1, subject2, role2, session, dyad_type

        Example:
            >>> dyads = loader.get_real_dyads(family='g01', dyad_type='therapist-patient')
        """
        # Validate task if provided
        if task:
            valid_tasks = self.get_intra_session_tasks()
            if task not in valid_tasks:
                logger.warning(f"Task '{task}' not in intra_session config")
                return []

        dyads = self.config.get("real_dyads", [])

        # Apply filters
        if family:
            dyads = [d for d in dyads if d["family"] == family]
        if session:
            ses_key = session if session.startswith("ses-") else f"ses-{session}"
            dyads = [d for d in dyads if d["session"] == ses_key]
        if dyad_type:
            dyads = [d for d in dyads if d["dyad_type"] == dyad_type]

        logger.debug(f"Found {len(dyads)} real dyads matching filters")
        return dyads

    def is_real_dyad(self, subject1: str, subject2: str, session: str) -> bool:
        """
        Check if a pair is a real therapeutic dyad.

        Args:
            subject1: First subject ID (with or without 'sub-' prefix)
            subject2: Second subject ID (with or without 'sub-' prefix)
            session: Session ID (e.g., 'ses-01')

        Returns:
            True if this is a real dyad, False otherwise

        Example:
            >>> loader.is_real_dyad('g01p01', 'g01p02', 'ses-01')
            True
        """
        s1 = subject1.replace("sub-", "")
        s2 = subject2.replace("sub-", "")
        ses = session if session.startswith("ses-") else f"ses-{session}"
        return (s1, s2, ses) in self._real_dyads_set

    def get_dyad_info(
        self, subject1: str, subject2: str, session: str
    ) -> Optional[Dict]:
        """
        Get full information for a specific dyad.

        Args:
            subject1: First subject ID
            subject2: Second subject ID
            session: Session ID

        Returns:
            Dyad dictionary if found, None otherwise
        """
        s1 = subject1.replace("sub-", "")
        s2 = subject2.replace("sub-", "")
        ses = session if session.startswith("ses-") else f"ses-{session}"

        for dyad in self.config.get("real_dyads", []):
            d_s1 = dyad["subject1"].replace("sub-", "")
            d_s2 = dyad["subject2"].replace("sub-", "")
            d_ses = dyad["session"]

            if d_ses == ses and {d_s1, d_s2} == {s1, s2}:
                return dyad

        return None

    # =========================================================================
    # Inter-Session Methods (all pairs for baseline comparison)
    # =========================================================================

    def get_all_session_pairs(
        self, task: Optional[str] = None
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Get all possible subject-session pairs across all sessions.

        Used for inter-session comparison to establish statistical baseline.

        Args:
            task: Optional task filter (e.g., 'therapy')

        Returns:
            List of pairs: [((subject1, session1), (subject2, session2)), ...]

        Example:
            >>> pairs = loader.get_all_session_pairs(task='therapy')
            >>> len(pairs)  # ~1275 for 51 subject-sessions
        """
        # Validate task
        if task:
            valid_tasks = self.get_inter_session_tasks()
            if task not in valid_tasks:
                logger.warning(f"Task '{task}' not in inter_session config")
                return []

        # Collect all subject-session combinations
        all_subject_sessions = []
        for family_id, family_data in self.config["families"].items():
            sessions_dict = family_data.get("sessions", {})
            for session, participants in sessions_dict.items():
                for participant in participants:
                    # Remove 'sub-' prefix if present for consistency
                    subj = participant.replace("sub-", "")
                    all_subject_sessions.append((subj, session))

        # Generate all unique pairs
        pairs = list(combinations(all_subject_sessions, 2))

        logger.info(f"Generated {len(pairs)} inter-session pairs")
        return pairs

    def get_all_session_pairs_with_real_flag(
        self, task: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all pairs with is_real_dyad flag for analysis.

        Returns:
            List of dicts: {subject1, session1, subject2, session2, is_real_dyad, dyad_info}
        """
        pairs = self.get_all_session_pairs(task=task)

        result = []
        for (s1, ses1), (s2, ses2) in pairs:
            # Only mark as real if same session
            is_real = ses1 == ses2 and self.is_real_dyad(s1, s2, ses1)
            dyad_info = self.get_dyad_info(s1, s2, ses1) if is_real else None

            result.append(
                {
                    "subject1": s1,
                    "session1": ses1,
                    "subject2": s2,
                    "session2": ses2,
                    "is_real_dyad": is_real,
                    "dyad_info": dyad_info,
                }
            )

        n_real = sum(1 for r in result if r["is_real_dyad"])
        logger.info(
            f"Generated {len(result)} pairs ({n_real} real, {len(result) - n_real} pseudo)"
        )
        return result

    # =========================================================================
    # Intra-Session Methods (real dyads only)
    # =========================================================================

    def get_intra_session_pairs(
        self,
        family: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get intra-session dyad pairs (real dyads within same session).

        This is an alias for get_real_dyads() for backward compatibility.

        Args:
            family: Optional family filter
            session: Optional session filter
            task: Optional task filter

        Returns:
            List of real dyad dictionaries
        """
        return self.get_real_dyads(family=family, session=session, task=task)

    # =========================================================================
    # Backward Compatibility - Old API
    # =========================================================================

    def get_inter_session_pairs(
        self, task: Optional[str] = None
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Get all inter-session dyad pairs (backward compatibility).

        Deprecated: Use get_all_session_pairs() instead.
        """
        return self.get_all_session_pairs(task=task)

    def get_intra_family_pairs(
        self,
        family: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]]:
        """
        Get intra-family dyad pairs (backward compatibility).

        Deprecated: Use get_real_dyads() instead for richer information.

        Returns pairs in old format: [((family, subject1, session), (family, subject2, session)), ...]
        """
        dyads = self.get_real_dyads(family=family, session=session, task=task)

        pairs = []
        for dyad in dyads:
            fam = dyad["family"]
            s1 = dyad["subject1"].replace("sub-", "")
            s2 = dyad["subject2"].replace("sub-", "")
            ses = dyad["session"]
            pairs.append(((fam, s1, ses), (fam, s2, ses)))

        return pairs

    def get_intra_family_method(self) -> str:
        """Backward compatibility alias for get_intra_session_method()."""
        return self.get_intra_session_method()

    def get_intra_family_tasks(self) -> List[str]:
        """Backward compatibility alias for get_intra_session_tasks()."""
        return self.get_intra_session_tasks()

    # =========================================================================
    # Epoching Configuration
    # =========================================================================

    def get_intra_session_method(self) -> str:
        """Get the epoching method for intra-session comparison."""
        return self.config["epoching"]["intra_session"]["method"]

    def get_inter_session_method(
        self, task: Optional[str] = None
    ) -> Union[str, Dict[str, str]]:
        """
        Get the epoching method for inter-session comparison.

        Args:
            task: Optional task name. If provided, returns method for that task.
                  If None, returns dict of all task->method mappings (new format)
                  or single method string (old format).

        Returns:
            If task is provided: method string for that task
            If task is None: dict {task: method} or single method string
        """
        inter_config = self.config["epoching"]["inter_session"]

        # New format: 'methods' dict per task
        if "methods" in inter_config:
            if task:
                return inter_config["methods"].get(task, "nsplit1")
            return inter_config["methods"]

        # Old format: single 'method' string (backward compatibility)
        if task:
            return inter_config.get("method", "nsplit120")
        return inter_config.get("method", "nsplit120")

    def get_inter_session_methods(self) -> Dict[str, str]:
        """
        Get all inter-session methods as a dict {task: method}.

        Returns:
            Dict mapping task name to epoching method
        """
        inter_config = self.config["epoching"]["inter_session"]

        # New format
        if "methods" in inter_config:
            return inter_config["methods"]

        # Old format: apply same method to all tasks
        method = inter_config.get("method", "nsplit120")
        tasks = inter_config.get("tasks", [])
        return {task: method for task in tasks}

    def get_intra_session_tasks(self) -> List[str]:
        """Get list of tasks for intra-session comparison."""
        return self.config["epoching"]["intra_session"]["tasks"]

    def get_inter_session_tasks(self) -> List[str]:
        """Get list of tasks for inter-session comparison."""
        return self.config["epoching"]["inter_session"]["tasks"]

    # =========================================================================
    # Family Methods
    # =========================================================================

    def get_families(self) -> List[str]:
        """Get list of all family IDs."""
        return list(self.config.get("families", {}).keys())

    def get_family_info(self, family: str) -> Optional[Dict]:
        """
        Get information about a specific family.

        Args:
            family: Family ID (e.g., 'g01')

        Returns:
            Dict with therapist, patients, and sessions info
        """
        return self.config.get("families", {}).get(family)

    def get_family_sessions(self, family: str) -> List[str]:
        """
        Get list of sessions for a specific family.

        Args:
            family: Family ID (e.g., 'g01')

        Returns:
            List of session IDs (e.g., ['ses-01', 'ses-02'])
        """
        family_data = self.get_family_info(family)
        if not family_data:
            logger.warning(f"Family '{family}' not found in config")
            return []
        return list(family_data.get("sessions", {}).keys())

    def get_therapist(self, family: str) -> Optional[str]:
        """Get therapist subject ID for a family."""
        family_data = self.get_family_info(family)
        if family_data:
            return family_data.get("therapist")
        return None

    def get_patients(self, family: str) -> Dict[str, str]:
        """Get patients for a family as {role: subject_id}."""
        family_data = self.get_family_info(family)
        if family_data:
            return family_data.get("patients", {})
        return {}

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict:
        """
        Get statistics about the dyad configuration.

        Returns:
            Dict with counts of families, sessions, real dyads, etc.
        """
        real_dyads = self.config.get("real_dyads", [])

        stats = {
            "n_families": len(self.get_families()),
            "n_real_dyads": len(real_dyads),
            "n_therapist_patient": sum(
                1 for d in real_dyads if d["dyad_type"] == "therapist-patient"
            ),
            "n_patient_patient": sum(
                1 for d in real_dyads if d["dyad_type"] == "patient-patient"
            ),
            "n_sessions": len(set(d["session"] for d in real_dyads)),
            "dyads_per_family": {},
            "dyads_per_session": {},
        }

        for family in self.get_families():
            family_dyads = [d for d in real_dyads if d["family"] == family]
            stats["dyads_per_family"][family] = len(family_dyads)

        for session in set(d["session"] for d in real_dyads):
            session_dyads = [d for d in real_dyads if d["session"] == session]
            stats["dyads_per_session"][session] = len(session_dyads)

        return stats
