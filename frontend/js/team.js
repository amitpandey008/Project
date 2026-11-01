/**
 * NeuroAdapt - Team Configuration
 * 
 * All team member names and roles are defined here.
 * Update this file to change team member information without
 * modifying any HTML structure.
 */
export const TEAM_MEMBERS = [
    {
        name: "AMIT SHARMA",
        role: "AI / ML Lead"
    },
    {
        name: "ABHISHEK SAXENA",
        role: "Backend Architect"
    }
];

/**
 * Get team member by index
 * @param {number} index - Team member index
 * @returns {object} Team member object
 */
export const getTeamMember = (index) => {
    return TEAM_MEMBERS[index] || { name: "Member " + (index + 1), role: "TBD" };
};

/**
 * Get all team members
 * @returns {array} Array of team member objects
 */
export const getAllTeamMembers = () => {
    return TEAM_MEMBERS;
};

/**
 * Add a new team member
 * @param {object} member - Member object with name and role
 */
export const addTeamMember = (member) => {
    TEAM_MEMBERS.push(member);
};
