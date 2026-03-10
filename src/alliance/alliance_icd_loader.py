"""
Alliance-ICD Data Loader.

Loads and merges alliance annotation states with ICD (Inter-Centroid Distance) 
data by epoch for correlation analysis.

Authors: Remy Ramadour
Date: November 2025
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class AllianceICDLoader:
    """Loads and merges alliance states with ICD data."""
    
    # Alliance state mapping
    ALLIANCE_STATES = {
        0: 'neutral',
        1: 'positive',
        -1: 'negative',
        2: 'split'  # Mixed positive and negative in same epoch
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize alliance-ICD loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path).config
        self.derivatives_path = Path(self.config['paths']['derivatives'])
        self.alliance_dir = self.config['output']['alliance_dir']
        
        # Cache for loaded data
        self._alliance_cache: Dict[str, pd.DataFrame] = {}
        self._icd_cache: Dict[str, pd.DataFrame] = {}
    
    def get_sessions_with_moi(self) -> Dict[str, List[str]]:
        """
        Discover which families and sessions have MOI annotations.
        
        Returns:
            Dict mapping family ID to list of session IDs with MOI
        """
        alliance_path = self.derivatives_path / self.alliance_dir
        sessions_with_moi = {}
        
        for family_dir in sorted(alliance_path.glob("sub-*shared")):
            family_id = family_dir.name.replace("sub-", "").replace("shared", "")
            sessions = []
            
            for session_dir in sorted(family_dir.glob("ses-*")):
                session_id = session_dir.name
                annotations_dir = session_dir / "annotations"
                if annotations_dir.exists() and list(annotations_dir.glob("*.tsv")):
                    sessions.append(session_id)
            
            if sessions:
                sessions_with_moi[family_id] = sessions
        
        return sessions_with_moi
    
    def load_alliance_states(
        self,
        group_id: str,
        session_id: str,
        method: str = 'nsplit'
    ) -> Dict[int, int]:
        """
        Load alliance states for a session.
        
        Args:
            group_id: Group ID (e.g., 'g01')
            session_id: Session ID (e.g., 'ses-01' or '01')
            method: Epoching method ('fixed', 'nsplit', 'sliding')
            
        Returns:
            Dict mapping epoch_id -> alliance state
                (0=neutral, 1=positive, -1=negative, 2=split)
        """
        # Normalize session_id format
        if not session_id.startswith('ses-'):
            session_id = f'ses-{session_id}'
        
        cache_key = f"{group_id}_{session_id}_{method}"
        
        if cache_key in self._alliance_cache:
            return self._alliance_cache[cache_key]
        
        # Load epoched annotations
        subject_dir = f"sub-{group_id}shared"
        annotations_subdir = self.config['output']['alliance_subdirs']['annotations']
        
        tsv_file = (
            self.derivatives_path /
            self.alliance_dir /
            subject_dir /
            session_id /
            annotations_subdir /
            f"sub-{group_id}_{session_id}_desc-alliance_annotations_epoched.tsv"
        )
        
        if not tsv_file.exists():
            raise FileNotFoundError(f"Epoched annotations not found: {tsv_file}")
        
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Parse epoch column
        epoch_col = f'epoch_{method}'
        if epoch_col not in df.columns:
            raise ValueError(f"Method '{method}' not found in annotations")
        
        df[epoch_col] = df[epoch_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Compute alliance states per epoch
        alliance_states = self._compute_alliance_states(df, epoch_col)
        
        self._alliance_cache[cache_key] = alliance_states
        return alliance_states
    
    def _compute_alliance_states(
        self,
        df: pd.DataFrame,
        epoch_col: str
    ) -> Dict[int, int]:
        """
        Compute alliance states for each epoch.
        
        Args:
            df: DataFrame with epoched annotations
            epoch_col: Name of epoch column
            
        Returns:
            Dict mapping epoch_id -> state
        """
        # Get all epochs with annotations
        epochs_with_annotations = set()
        for epoch_list in df[epoch_col]:
            if isinstance(epoch_list, list):
                epochs_with_annotations.update(epoch_list)
        
        if not epochs_with_annotations:
            return {}
        
        max_epoch = max(epochs_with_annotations)
        alliance_states = {i: 0 for i in range(max_epoch + 1)}
        
        for epoch_id in epochs_with_annotations:
            overlapping = df[df[epoch_col].apply(lambda x: epoch_id in x if isinstance(x, list) else False)]
            
            alliance_values = overlapping['alliance'].dropna()
            has_positive = (alliance_values == 1).any()
            has_negative = (alliance_values == -1).any()
            
            if has_positive and has_negative:
                alliance_states[epoch_id] = 2  # Split
            elif has_positive:
                alliance_states[epoch_id] = 1  # Positive
            elif has_negative:
                alliance_states[epoch_id] = -1  # Negative
            # else: 0 (neutral)
        
        return alliance_states
    
    def load_icd_data(
        self,
        dyad_type: str = 'intra',
        task: str = 'therapy',
        method: str = 'nsplit120'
    ) -> pd.DataFrame:
        """
        Load ICD data from CSV file.
        
        Args:
            dyad_type: 'intra' for intra-family, 'inter' for inter-session
            task: Task name ('therapy' or 'restingstate')
            method: Epoching method with parameters
            
        Returns:
            DataFrame with epoch_id as index and dyad columns
        """
        cache_key = f"{dyad_type}_{task}_{method}"
        
        if cache_key in self._icd_cache:
            return self._icd_cache[cache_key]
        
        dppa_dir = self.derivatives_path / 'dppa'
        
        if dyad_type == 'intra':
            # Intra-family uses sliding method
            icd_file = dppa_dir / 'intra_family' / f'intra_family_icd_task-{task}_method-{method}.csv'
        else:
            icd_file = dppa_dir / 'inter_session' / f'inter_session_icd_task-{task}_method-{method}.csv'
        
        if not icd_file.exists():
            raise FileNotFoundError(f"ICD file not found: {icd_file}")
        
        df = pd.read_csv(icd_file)
        df = df.set_index('epoch_id')
        
        self._icd_cache[cache_key] = df
        return df
    
    def get_valid_dyads(
        self,
        sessions_with_moi: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Get lists of valid real and pseudo dyads from the inter_session file.
        
        The inter_session file contains ALL pairs including:
        - Real dyads: same family, same session (e.g., g01p02_ses-01_vs_g01p01_ses-01)
        - Pseudo-dyads: different families (e.g., g01p02_ses-01_vs_g03p02_ses-01)
        - Cross-session same-family: same family, different sessions (excluded)
        
        Args:
            sessions_with_moi: Dict mapping family -> sessions with MOI.
                If None, will be discovered.
                
        Returns:
            Tuple of (valid_real_dyads, valid_pseudo_dyads) as column names
        """
        if sessions_with_moi is None:
            sessions_with_moi = self.get_sessions_with_moi()
        
        valid_real = []
        valid_pseudo = []
        
        # Load inter-session ICD (contains ALL pairs in nsplit120)
        try:
            inter_df = self.load_icd_data('inter', 'therapy', 'nsplit120')
            
            for col in inter_df.columns:
                # Format: g01p02_ses-01_vs_g03p02_ses-01
                parts = col.split('_vs_')
                p1_full = parts[0]
                p2_full = parts[1]
                
                p1_parts = p1_full.rsplit('_', 1)
                p1, s1 = p1_parts[0], p1_parts[1]
                
                p2_parts = p2_full.rsplit('_', 1)
                p2, s2 = p2_parts[0], p2_parts[1]
                
                f1, f2 = p1[:3], p2[:3]
                
                # Check if both sessions have MOI
                f1_valid = f1 in sessions_with_moi and s1 in sessions_with_moi[f1]
                f2_valid = f2 in sessions_with_moi and s2 in sessions_with_moi[f2]
                
                if not (f1_valid and f2_valid):
                    continue  # Skip if either session doesn't have MOI
                
                if f1 == f2 and s1 == s2:
                    # Same family, same session = REAL dyad
                    valid_real.append(col)
                elif f1 != f2:
                    # Different families = PSEUDO-dyad
                    valid_pseudo.append(col)
                # else: same family, different sessions = skip (not useful for this analysis)
                    
        except FileNotFoundError:
            logger.warning("Inter-session ICD file not found")
        
        return valid_real, valid_pseudo
    
    def merge_alliance_icd(
        self,
        dyad_column: str,
        icd_df: pd.DataFrame,
        dyad_type: str = 'real'
    ) -> pd.DataFrame:
        """
        Merge alliance states with ICD data for a specific dyad.
        
        All dyads (real and pseudo) use the same format from inter_session file:
        Format: g01p02_ses-01_vs_g03p02_ses-01
        
        For real dyads: uses the family's alliance (same family for both)
        For pseudo-dyads: uses first family's alliance (lexicographic order)
        
        Args:
            dyad_column: Column name from ICD data
            icd_df: ICD DataFrame
            dyad_type: 'real' or 'pseudo'
            
        Returns:
            DataFrame with columns: epoch_id, icd, alliance_state, alliance_label
        """
        # Parse dyad info - unified format: g01p02_ses-01_vs_g03p02_ses-01
        parts = dyad_column.split('_vs_')
        p1_full = parts[0]
        p2_full = parts[1]
        
        p1_parts = p1_full.rsplit('_', 1)
        p1, s1 = p1_parts[0], p1_parts[1]
        
        p2_parts = p2_full.rsplit('_', 1)
        p2, s2 = p2_parts[0], p2_parts[1]
        
        f1, f2 = p1[:3], p2[:3]
        
        # Determine which family/session to use for alliance direction
        if dyad_type == 'real':
            # Real dyad: same family, same session
            family, session = f1, s1
        else:
            # Pseudo-dyad: use first family (lexicographic) for alliance direction
            if f1 < f2:
                family, session = f1, s1
            else:
                family, session = f2, s2
        
        # Load alliance states
        try:
            alliance_states = self.load_alliance_states(family, session, method='nsplit')
        except FileNotFoundError:
            logger.warning(f"Alliance not found for {family}/{session}")
            return pd.DataFrame()
        
        # Extract ICD values
        icd_values = icd_df[dyad_column].dropna()
        
        # Build merged DataFrame
        records = []
        for epoch_id, icd_value in icd_values.items():
            state = alliance_states.get(epoch_id, 0)
            records.append({
                'epoch_id': epoch_id,
                'icd': icd_value,
                'alliance_state': state,
                'alliance_label': self.ALLIANCE_STATES[state],
                'dyad': dyad_column,
                'dyad_type': dyad_type,
                'family': family,
                'session': session
            })
        
        return pd.DataFrame(records)
    
    def load_all_merged_data(
        self,
        sessions_with_moi: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Load and merge all valid dyad data with alliance states.
        
        Uses ONLY the inter_session file (nsplit120) which contains:
        - Real dyads: same family, same session
        - Pseudo-dyads: different families
        
        Args:
            sessions_with_moi: Dict mapping family -> sessions with MOI
            
        Returns:
            DataFrame with all merged data
        """
        if sessions_with_moi is None:
            sessions_with_moi = self.get_sessions_with_moi()
        
        valid_real, valid_pseudo = self.get_valid_dyads(sessions_with_moi)
        
        all_data = []
        
        # Load the single source file (inter_session with nsplit120)
        try:
            inter_df = self.load_icd_data('inter', 'therapy', 'nsplit120')
        except FileNotFoundError:
            logger.error("Inter-session ICD file not found")
            return pd.DataFrame()
        
        # Process real dyads (from inter_session file)
        logger.info(f"Processing {len(valid_real)} real dyads...")
        for dyad in valid_real:
            merged = self.merge_alliance_icd(dyad, inter_df, 'real')
            if not merged.empty:
                all_data.append(merged)
        
        # Process pseudo-dyads (from inter_session file)
        logger.info(f"Processing {len(valid_pseudo)} pseudo-dyads...")
        for dyad in valid_pseudo:
            merged = self.merge_alliance_icd(dyad, inter_df, 'pseudo')
            if not merged.empty:
                all_data.append(merged)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(result)} epoch-dyad observations")
        
        return result
