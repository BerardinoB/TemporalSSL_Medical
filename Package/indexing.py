import itertools
from typing import List

class SubjectKeys:
    MRI_AND_LABEL = 'MRI_AND_LABEL'
    VBM = 'VBM'
    ANTS_DEFORMS = 'ANTS_DEFORMS'
    XFM_DEFORMS = 'XFM_DEFORMS'
    CLINICAL = 'CLINICAL'
    USER = 'USER'

class UserKeys:
    AGE = 'AGE'
    VISIT_WEEK = 'VISIT_WEEK'
    STD_MRI_VISIT_LABEL = 'STD_MRI_VISIT_LABEL' # Standardized MRI Visit names
    NEAREST_STD_MRI_VISIT_LABEL_BY_VISIT_WEEK = 'NEAREST_STD_MRI_VISIT_LABEL_BY_VISIT_WEEK'
    MS_SYMPTOM_DURATION = 'MS_SYMPTOM_DURATION'
    MRI_ACTIVITY = 'MRI_ACTIVITY'

    @staticmethod
    def generate_idp_visit_week_key(day: str, measure: str):
        assert day in ClinicalKeys.CDP.Days.ALL
        assert measure in ClinicalKeys.CDP.Measure.ALL

        return f"CDP_{day}d_{measure}_IDP_Visit_Week"

    @staticmethod
    def generate_idp_visit_week_key_list(days: List[str], measures: List[str]):
        assert isinstance(days, list)
        assert isinstance(measures, list)

        return [ UserKeys.generate_idp_visit_week_key(day, measure)
                 for (day, measure) in itertools.product(days, measures) ]

class ClinicalKeys:
    VISIT_LABEL = 'VISIT_Label'
    VISIT_DATE = 'VISIT_Date'
    VISIT_DAY = 'VISIT_Day'
    LORIS_ID = 'SUBJECT_LORIS_ID_Number'
    TRIAL_NAME = 'SUBJECT_Trial_Name'

    class Reference:
        REFERENCE_DATE = 'REFERENCE_Date'
        REFERENCE_AGE = 'REFERENCE_Age_yrs'
        REFERENCE_EDSS_SCORE = 'REFERENCE_EDSS_Score'
        REFERENCE_T25FW_MEAN = 'REFERENCE_T25FW_Mean_sec'
        REFERENCE_NINE_HPT_DOMHAND_MEAN = 'REFERENCE_9HPT_DomHand_Mean_sec'
        REFERENCE_NINE_HPT_NONDOMHAND_MEAN = 'REFERENCE_9HPT_NonDomHand_Mean_sec'

        ALL = [ REFERENCE_DATE, REFERENCE_AGE,
                REFERENCE_EDSS_SCORE, REFERENCE_T25FW_MEAN,
                REFERENCE_NINE_HPT_DOMHAND_MEAN, REFERENCE_NINE_HPT_NONDOMHAND_MEAN,
              ]

    class Subject:
        # Categorical
        TRIAL_ARM = 'SUBJECT_Trial_Arm'
        SEX = 'SUBJECT_Sex'
        DOMINANT_HAND = 'SUBJECT_Dominant_Hand'
        RACE = 'SUBJECT_Race'
        COUNTRY = 'SUBJECT_Country'
        OBESITY = 'SUBJECT_Obesity'
        SMOKING = 'SUBJECT_Smoking'
        MS_TYPE = 'SUBJECT_Screening_Visit_MS_Type'

        # Binary
        HYPERTENSION = 'SUBJECT_Hypertension'
        DIABETES_MELLITUS = 'SUBJECT_Diabetes_Mellitus'
        HYPERLIPIDAEMIA = 'SUBJECT_Hyperlipidaemia'
        CORONARY_HEART_DISEASE = 'SUBJECT_Coronary_Heart_Disease'

        # Numerical
        SCREENING_HEIGHT = 'SUBJECT_Screening_Height_cm'
        SCREENING_WEIGHT = 'SUBJECT_Screening_Weight_kg'

        # Date
        BASELINE_VISIT_DATE = 'SUBJECT_Baseline_Visit_Date'
        BIRTH_DATE = 'SUBJECT_Birth_Date'
        MS_SYMPTOM_ONSET_DATE = 'SUBJECT_MS_Symptom_Onset_Date'

        ALL = [ TRIAL_ARM, SEX, DOMINANT_HAND, RACE, COUNTRY, OBESITY, SMOKING, MS_TYPE,
                HYPERTENSION, DIABETES_MELLITUS, HYPERLIPIDAEMIA, CORONARY_HEART_DISEASE,
                SCREENING_HEIGHT, SCREENING_WEIGHT,
                BASELINE_VISIT_DATE, BIRTH_DATE, MS_SYMPTOM_ONSET_DATE,
              ]

    class Clinical:
        """
            EDSS: Expanded Disability Status Scale
            FSS: Functional Systems Score
            T25FW: Timed 25 Foot Walk
            9HPT: 9-Hole Peg Test
        """
        EDSS_SCORE_OBSERVED = 'CLINICAL_EDSS_Score_Observed'
        FSS_BOWEL_BLADDER = 'CLINICAL_FSS_Bowel_Bladder'
        FSS_BRAINSTEM = 'CLINICAL_FSS_Brainstem'
        FSS_CEREBELLAR = 'CLINICAL_FSS_Cerebellar'
        FSS_CEREBRAL_MENTAL = 'CLINICAL_FSS_Cerebral_Mental'
        FSS_PYRAMIDAL = 'CLINICAL_FSS_Pyramidal'
        FSS_SENSORY = 'CLINICAL_FSS_Sensory'
        FSS_VISUAL = 'CLINICAL_FSS_Visual'
        T25FW_MEAN = 'CLINICAL_T25FW_Mean_sec'
        NINE_HPT_DOMHAND_MEAN = 'CLINICAL_9HPT_DomHand_Mean_sec'
        NINE_HPT_NONDOMHAND_MEAN = 'CLINICAL_9HPT_NonDomHand_Mean_sec'

        ALL = [ EDSS_SCORE_OBSERVED,
                FSS_BOWEL_BLADDER, FSS_BRAINSTEM, FSS_CEREBELLAR, FSS_CEREBRAL_MENTAL,
                FSS_PYRAMIDAL, FSS_SENSORY, FSS_VISUAL,
                T25FW_MEAN,
                NINE_HPT_DOMHAND_MEAN, NINE_HPT_NONDOMHAND_MEAN,
              ]

    class MRI:
        """
            NBV: Normalized Brain Volume
        """
        LESION_GAD_CONSENSUS_COUNT = 'MRI_Lesion_Gad_Consensus_Count'
        LESION_NEWENLT2_COUNT = 'MRI_Lesion_NewEnlT2_Count'
        LESION_NEWT1_COUNT = 'MRI_Lesion_NewT1_Count'
        LESION_T2_VOL = 'MRI_Lesion_T2_Vol_ml'
        LESION_T1_VOL = 'MRI_Lesion_T1_Vol_ml'
        VOLUME_NBV = 'MRI_Volume_NBV_ml'

        ALL = [ LESION_GAD_CONSENSUS_COUNT, LESION_NEWENLT2_COUNT, LESION_NEWT1_COUNT,
                LESION_T2_VOL, LESION_T1_VOL,
                VOLUME_NBV,
              ]

    class CDP:
        """
            CDP: Confirmed Disability Progression, when disability progression has been
                 confirmed to be sustained for x days
        """
        class Days:
            D010 = '010'
            D019 = '019'
            D038 = '038'
            D076 = '076'
            D084 = '084' # Only in 20210610 CSVs, reintroduced in 20220202 CSVs
            D114 = '114' # Only in 20210610 CSVs
            D152 = '152'
            D168 = '168' # Only in 20210610 CSVs, reintroduced in 20220202 CSVs
            D228 = '228' # Only in 20210922 CSVs and later
            D304 = '304' # Only in 20210922 CSVs and later

            ALL = [ D010, D019, D038, D076, D152 ]

        class Measure:
            EDSS = 'EDSS'
            EDSS_PLUS_A = 'EDSSPlus_A'
            EDSS_PLUS_B = 'EDSSPlus_B'
            T25FW = 'T25FW'
            NINE_HPT_DOMHAND = '9HPT_DomHand'
            NINE_HPT_NONDOMHAND = '9HPT_NonDomHand'

            ALL = [ EDSS, EDSS_PLUS_A, EDSS_PLUS_B,
                    T25FW,
                    NINE_HPT_DOMHAND, NINE_HPT_NONDOMHAND,
                  ]

        class Info:
            """
                IDP: Initial Disability Progression, when disability progression has been
                     initially observed (but may recede)
                Reference_to_CDP_days: Number of days from reference visit to IDP (not actually CDP)
                IDP_to_Confirmation_Days: Number of days from IDP visit to confirmation visit
            """
            REFERENCE_TO_CDP_DAYS = 'Reference_to_CDP_days'
            IDP_TO_CONFIRMATION_DAYS = 'IDP_to_Confirmation_days'
            STATUS_FOR_TRIAL = 'Status_for_Trial'
            STATUS_AT_VISIT = 'Status_at_Visit'
            IDP_VISIT_LABEL = 'IDP_Visit_Label'
            IDP_VISIT_DATE = 'IDP_Visit_Date'
            IDP_TRIGGER_SCORE = 'IDP_Trigger_Score' # EDSS only
            IDP_TRIGGER_SCORE_CHANGE = 'IDP_Trigger_Change' # EDSS only
            IDP_TRIGGER_TIME = 'IDP_Trigger_Time_sec' # For T25FW and 9HPT
            IDP_TRIGGER_TIME_CHANGE = 'IDP_Trigger_Change_sec' # For T25FW and 9HPT

            ALL = [ REFERENCE_TO_CDP_DAYS, IDP_TO_CONFIRMATION_DAYS,
                    STATUS_FOR_TRIAL, STATUS_AT_VISIT,
                    IDP_VISIT_LABEL, IDP_VISIT_DATE,
                    IDP_TRIGGER_SCORE, IDP_TRIGGER_SCORE_CHANGE,
                    IDP_TRIGGER_TIME, IDP_TRIGGER_TIME_CHANGE,
                  ]

        @classmethod
        def generate_key(cls, day: str, measure: str, info: str):
            assert day in cls.Days.ALL
            assert measure in cls.Measure.ALL
            assert info in cls.Info.ALL

            return f"CDP_{day}d_{measure}_{info}"

        @classmethod
        def generate_key_list(cls, days: List[str], measures: List[str], infos: List[str]):
            assert isinstance(days, list)
            assert isinstance(measures, list)
            assert isinstance(infos, list)

            return [ cls.generate_key(day, measure, info) for (day, measure, info)
                     in itertools.product(days, measures, infos) ]

    ALL = ( [ VISIT_LABEL, VISIT_DATE,  VISIT_DAY, LORIS_ID, TRIAL_NAME ]
            + Reference.ALL + Subject.ALL + Clinical.ALL + MRI.ALL
            + CDP.generate_key_list([ CDP.Days.D010,
                                      CDP.Days.D019,
                                      CDP.Days.D038,
                                      CDP.Days.D076,
                                      CDP.Days.D152, ],
                                     CDP.Measure.ALL, CDP.Info.ALL)
          )
