document.addEventListener('DOMContentLoaded', function() {
    // --- Element Selectors ---
    const analyzeBtn = document.getElementById('analyzeBtn');
    const symptomsInput = document.getElementById('symptomsInput');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessageEl = document.getElementById('errorMessage');

    // --- Result Display Elements ---
    const emergencyWarning = document.getElementById('emergencyWarning');
    const severityProgress = document.getElementById('severityProgress');
    const severityText = document.getElementById('severityText');
    const severityBadge = document.getElementById('severityBadge');
    const specialistName = document.getElementById('specialistName');
    const specialistDescription = document.getElementById('specialistDescription');
    const specialistIcon = document.getElementById('specialistIcon');
    const urgencyTextEl = document.getElementById('urgencyText');
    const urgencyDescriptionEl = document.getElementById('urgencyDescription');
    const urgencyIconEl = document.getElementById('urgencyIcon');

    // --- Detailed Analysis Elements ---
    const detailedAnalysisContainer = document.getElementById('detailedAnalysis');
    const originalComplaintTextEl = document.getElementById('originalComplaintText');
    const enhancedComplaintTextEl = document.createElement('p');
    enhancedComplaintTextEl.className = 'mb-2 text-sm text-gray-400';
    const processedComplaintTextEl = document.getElementById('processedComplaintText');
    const geminiExplanationEl = document.createElement('div');
    geminiExplanationEl.className = 'mt-4 prose prose-sm max-w-none text-gray-200';
    const conditionsList = document.getElementById('conditionsList');

    // Backend API URL
    const backendUrl = 'http://127.0.0.1:5000/predict'; // Your Flask server address

    // --- Data Mappings (Translated to English) ---

    // Note: Keys must be lowercase to match backend output and script logic.
    const specialistsData = {
        'general practitioner': { icon: 'fas fa-stethoscope', description: 'General specialist for diagnosing and treating a wide range of diseases.' },
        'neurologist': { icon: 'fas fa-brain', description: 'Treats nervous system disorders, headaches, and sleep issues.' },
        'cardiologist': { icon: 'fas fa-heartbeat', description: 'Specialist in heart and blood vessel diseases.' },
        'gastroenterologist': { icon: 'fas fa-pills', description: 'Specializes in diseases of the gastrointestinal tract.' },
        'pulmonologist': { icon: 'fas fa-lungs', description: 'Specialist in lung and respiratory diseases.' },
        'urologist': { icon: 'fas fa-toilet-paper', description: 'Treats diseases of the urinary system.' },
        'endocrinologist': { icon: 'fas fa-cogs', description: 'Treats hormonal disorders and metabolic diseases.' },
        'dermatologist': { icon: 'fas fa-hand-sparkles', description: 'Specialist in skin, hair, and nail diseases.' },
        'ophthalmologist': { icon: 'fas fa-eye', description: 'Doctor specializing in eye diseases.' },
        'ent (otolaryngologist)': { icon: 'fas fa-head-side-cough', description: 'Specialist in ear, nose, and throat diseases (otolaryngologist).' },
        'surgeon': { icon: 'fas fa-band-aid', description: 'Specialist who treats diseases requiring surgical intervention.' },
        'traumatologist': { icon: 'fas fa-procedures', description: 'Treats injuries, fractures, and musculoskeletal system diseases.' },
        'pediatrician': { icon: 'fas fa-baby', description: 'Specialist in children\'s health and diseases.' },
        'gynecologist': { icon: 'fas fa-venus', description: 'Specialist in women\'s health and the reproductive system.' },
        'dentist': { icon: 'fas fa-tooth', description: 'Doctor who treats diseases of the teeth and oral cavity.' },
        'psychotherapist': { icon: 'fas fa-comment-medical', description: 'Specialist in mental health and emotional issues.' },
        'nephrologist': { icon: 'fas fa-kidneys', description: 'Specialist in kidney diseases.' },
        'allergist': { icon: 'fas fa-allergies', description: 'Specialist in allergic reactions and diseases.' },
        'rheumatologist': { icon: 'fas fa-joint', description: 'Specialist in joint and connective tissue diseases.' },
        'phlebologist': { icon: 'fas fa-water', description: 'Specialist in vein (blood vessel) diseases.' },
        'proctologist': { icon: 'fas fa-poop', description: 'Specialist in diseases of the rectum and adjacent organs.' },
        'undetermined': { icon: 'fas fa-question-circle', description: 'The specialist could not be determined. Please describe your symptoms in more detail.' },
        // Combined specialists
        'surgeon/traumatologist': { icon: 'fas fa-procedures', description: 'Helps with injuries, fractures, and situations requiring surgery.' },
        'pediatrician/dermatologist': { icon: 'fas fa-baby', description: 'Specialist in children\'s skin diseases.' },
        'general practitioner/neurologist': { icon: 'fas fa-stethoscope', description: 'Assists with general therapeutic and nervous system issues.' },
        'surgeon/phlebologist': { icon: 'fas fa-band-aid', description: 'Specialist in vein diseases and surgical interventions.' },
        'pediatrician/surgeon': { icon: 'fas fa-baby', description: 'Specialist in pediatric surgery.' },
        'dentist/neurologist': { icon: 'fas fa-tooth', description: 'Specialist in dental and maxillofacial neurological issues.' },
        'ent/general practitioner': { icon: 'fas fa-head-side-cough', description: 'Specialist in ENT and general therapeutic issues.' },
        'general practitioner/proctologist': { icon: 'fas fa-stethoscope', description: 'Assists with general therapeutic and proctological issues.' },
        'neurologist/surgeon': { icon: 'fas fa-brain', description: 'Specialist in nervous system diseases requiring surgical treatment.' },
        'cardiologist/general practitioner': { icon: 'fas fa-heartbeat', description: 'Assists with cardiovascular and general therapeutic issues.' },
        'neurologist/traumatologist': { icon: 'fas fa-brain', description: 'Specialist in nervous system injuries and traumatological issues.' },
        'gastroenterologist/general practitioner': { icon: 'fas fa-pills', description: 'Assists with gastrointestinal and general therapeutic issues.' }
    };

    // Note: Keys must match the exact output from the Python backend.
    const urgencyMapping = {
        'Red': { key: 'immediate', severityScore: 95, badgeColor: 'bg-red-600 text-red-100', text: 'Critical' },
        'Orange': { key: 'urgent', severityScore: 75, badgeColor: 'bg-orange-600 text-orange-100', text: 'High' },
        'Yellow': { key: 'soon', severityScore: 50, badgeColor: 'bg-yellow-600 text-yellow-100', text: 'Medium' },
        'Green': { key: 'routine', severityScore: 25, badgeColor: 'bg-green-600 text-green-100', text: 'Low' },
        'Undetermined': { key: 'unknown', severityScore: 0, badgeColor: 'bg-gray-600 text-gray-100', text: 'Unknown' }
    };

    const urgencyLevelsDetails = {
        'immediate': {
            text: 'Immediate Care',
            description: 'Potentially life-threatening. Immediate medical attention is required!',
            icon: 'fas fa-ambulance text-red-500'
        },
        'urgent': {
            text: 'Urgent',
            description: 'It is recommended to see a doctor within 24 hours.',
            icon: 'fas fa-exclamation-triangle text-orange-500'
        },
        'soon': {
            text: 'Soon',
            description: 'A visit to the doctor should be planned in the coming days.',
            icon: 'fas fa-clock text-yellow-500'
        },
        'routine': {
            text: 'Routine',
            description: 'You can schedule a routine visit to the doctor.',
            icon: 'fas fa-calendar-check text-green-500'
        },
        'unknown': {
            text: 'Unknown',
            description: 'The urgency level could not be determined. Please provide more details about your symptoms.',
            icon: 'fas fa-question-circle text-gray-500'
        }
    };


    analyzeBtn.addEventListener('click', async function() {
        const symptoms = symptomsInput.value.trim();

        if (!symptoms) {
            showError('Please describe your symptoms.');
            return;
        }

        setLoadingState(true);

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ complaint: symptoms }),
            });

            if (!response.ok) {
                let errorMsg = `Server error: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg; // Use the English error key
                } catch (e) { /* Ignore JSON parsing error */ }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            displayResults(data, symptoms);

        } catch (error) {
            console.error('Error during analysis:', error);
            showError(`Error: ${error.message}. Please try again or check if the server is running.`);
            resultsSection.classList.add('hidden');
        } finally {
            setLoadingState(false);
        }
    });

    function displayResults(data, originalSymptoms) {
        resultsSection.classList.remove('hidden');
        errorSection.classList.add('hidden'); // Hide error on success

        // --- Display Urgency Level ---
        const backendUrgencyKey = data.predicted_urgency || 'Undetermined';
        const urgencyInfo = urgencyMapping[backendUrgencyKey] || urgencyMapping['Undetermined'];

        severityProgress.style.width = urgencyInfo.severityScore + '%';
        severityText.textContent = urgencyInfo.text; // For the badge
        severityBadge.className = `px-4 py-1 rounded-full text-sm font-medium ${urgencyInfo.badgeColor}`;
        severityBadge.classList.toggle('hidden', urgencyInfo.severityScore === 0);

        const urgencyDetails = urgencyLevelsDetails[urgencyInfo.key];
        urgencyTextEl.textContent = urgencyDetails.text; // For the dedicated block
        urgencyDescriptionEl.textContent = urgencyDetails.description;
        urgencyIconEl.className = `${urgencyDetails.icon} text-4xl mb-2`;

        emergencyWarning.classList.toggle('hidden', urgencyInfo.key !== 'immediate');

        // --- Display Recommended Specialist ---
        const backendSpecialistRaw = data.predicted_specialist || 'Undetermined';
        const backendSpecialistLC = backendSpecialistRaw.toLowerCase(); // For matching keys
        const specialistInfo = specialistsData[backendSpecialistLC] || specialistsData['undetermined'];

        // Capitalize each word for display
        let specialistDisplayName = backendSpecialistRaw.split('/')
            .map(s => s.charAt(0).toUpperCase() + s.slice(1)).join('/');

        specialistName.textContent = specialistDisplayName;
        specialistDescription.textContent = specialistInfo.description;
        specialistIcon.className = `${specialistInfo.icon} text-4xl mb-2`;

        // --- Display Detailed Analysis Section ---
        detailedAnalysisContainer.innerHTML = ''; // Clear previous content

        originalComplaintTextEl.textContent = `Original Complaint: ${data.complaint_original || originalSymptoms}`;
        detailedAnalysisContainer.appendChild(originalComplaintTextEl);

        if (data.complaint_enhanced && data.complaint_enhanced !== "Enhancement not applied or no changes made.") {
            enhancedComplaintTextEl.textContent = `AI Enhanced Complaint: ${data.complaint_enhanced}`;
            detailedAnalysisContainer.appendChild(enhancedComplaintTextEl);
        }

        processedComplaintTextEl.textContent = `Processed for Model: ${data.complaint_processed_for_model || 'no data'}`;
        detailedAnalysisContainer.appendChild(processedComplaintTextEl);

        // --- Display Gemini's Enriched Response ---
        geminiExplanationEl.innerHTML = ''; // Clear
        let contentAdded = false;

        if (data.gemini_explanation && data.gemini_explanation !== "Explanation could not be generated.") {
            addDetailedSection('AI Explanation:', [data.gemini_explanation], 'p');
            contentAdded = true;
        }

        if (data.gemini_follow_up_questions && data.gemini_follow_up_questions.length > 0) {
             addDetailedSection('Follow-up Questions:', data.gemini_follow_up_questions, 'ul');
             contentAdded = true;
        }

        if (data.gemini_general_advice) {
             addDetailedSection('General Advice:', [data.gemini_general_advice], 'p');
             contentAdded = true;
        }

        if (contentAdded) {
            detailedAnalysisContainer.appendChild(geminiExplanationEl);
        }

        // --- Display "Possible Conditions" (Demo) ---
        conditionsList.innerHTML = ''; // Clear old list
        const noConditionsMsg = document.createElement('p');
        noConditionsMsg.className = 'text-sm text-gray-400';
        noConditionsMsg.textContent = 'Information on possible conditions is not yet available.';
        conditionsList.appendChild(noConditionsMsg);
    }

    function addDetailedSection(title, items, listType) {
        const titleEl = document.createElement('h4');
        titleEl.className = 'font-semibold text-cyan-400 mt-3 mb-1';
        titleEl.textContent = title;
        geminiExplanationEl.appendChild(titleEl);

        if (listType === 'ul') {
            const listEl = document.createElement('ul');
            listEl.className = 'list-disc list-inside ml-4';
            items.forEach(itemText => {
                const li = document.createElement('li');
                li.textContent = itemText;
                listEl.appendChild(li);
            });
            geminiExplanationEl.appendChild(listEl);
        } else { // 'p'
            const p = document.createElement('p');
            p.textContent = items[0];
            geminiExplanationEl.appendChild(p);
        }
    }


    function setLoadingState(isLoading) {
        analyzeBtn.disabled = isLoading;
        if (isLoading) {
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...';
        } else {
            analyzeBtn.innerHTML = '<i class="fas fa-redo mr-2"></i> Analyze Again';
        }
    }

    function showError(message) {
        errorMessageEl.textContent = message;
        errorSection.classList.remove('hidden');
    }
});